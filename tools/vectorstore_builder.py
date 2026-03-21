from tools.s3_utils import get_secret
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import boto3
import json
import hashlib

# --- Load API Key ---
def get_openai_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("❌ OPENAI_API_KEY is not set. Please check your .env file or Streamlit secrets.")
    return key

# --- Build and Save Combined Vectorstore ---
def build_vectorstore(
    pdf_path="docs/EmployeeHandbook.pdf",
    docx_path="docs/innovim_onboarding.docx",
    index_path="faiss_index",
    api_key=None
):
    print("🔍 Checking for existing FAISS index...")
    index_file = Path(index_path) / "index.faiss"

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

    if index_file.exists():
        print(f"✅ Existing vectorstore found at '{index_path}/'. Loading...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("🚧 No index found. Building new vectorstore...")

    # --- Load PDF ---
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        doc.metadata["source"] = "employee_handbook"

    # --- Load DOCX ---
    docx_loader = UnstructuredWordDocumentLoader(docx_path)
    docx_docs = docx_loader.load()
    for doc in docx_docs:
        doc.metadata["source"] = "orientation_guide"

    # --- Combine and Split ---
    all_docs = pdf_docs + docx_docs
    print(f"📄 Loaded {len(all_docs)} total documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    docs = splitter.split_documents(all_docs)
    print(f"✂️ Split into {len(docs)} chunks")

    # --- Embed and Save ---
    print("💾 Saving FAISS index...")
    Path(index_path).mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)

    print(f"✅ Vectorstore built and saved to '{index_path}/'")
    return vectorstore

# --- Optional CLI ---
if __name__ == "__main__":
    build_vectorstore(index_path="faiss_index_hr_combined")


def rebuild_vectorstore_from_docs(docs_path="docs", faiss_path="faiss_index"):
    docs_path = Path(docs_path)
    all_docs = []

    for doc_file in docs_path.glob("*"):
        if doc_file.suffix == ".pdf":
            loader = PyPDFLoader(str(doc_file))
        elif doc_file.suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(doc_file))
        else:
            continue
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

    os.makedirs("faiss_index", exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return len(all_docs), len(chunks)


def rebuild_vectorstore_from_s3():
    """
    Full rebuild — downloads all docs from S3, embeds them,
    saves locally, and uploads new index back to S3.
    Returns (doc_count, chunk_count)
    """
    import streamlit as st
    print("🔄 Starting full vectorstore rebuild from S3...")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION")
        )
        docs_bucket = get_secret("S3_DOCS_BUCKET")
        index_bucket = get_secret("S3_INDEX_BUCKET")
    except Exception as e:
        print(f"❌ Could not connect to S3: {e}")
        return 0, 0

    response = s3.list_objects_v2(Bucket=docs_bucket)
    if "Contents" not in response:
        print("❌ No documents found in S3.")
        return 0, 0

    all_docs = []
    doc_count = 0

    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith((".pdf", ".docx")):
            continue

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf" if key.endswith(".pdf") else ".docx"
            ) as tmp_file:
                s3.download_file(docs_bucket, key, tmp_file.name)
                print(f"⬇️ Downloaded: {key}")

                clean_name = key.replace(".pdf", "").replace(".docx", "")
                clean_name = clean_name.replace(" ", "_").replace("-", "_").lower()

                if key.endswith(".pdf"):
                    from tools.loaders import enrich_pdf_chunks
                    loaded_docs = enrich_pdf_chunks(tmp_file.name)
                    for doc in loaded_docs:
                        doc.metadata["source"] = clean_name
                else:
                    from tools.loaders import chunk_docx_with_metadata
                    loaded_docs = chunk_docx_with_metadata(tmp_file.name)
                    for doc in loaded_docs:
                        doc.metadata["source"] = clean_name

                print(f"📄 Loaded {len(loaded_docs)} pages from {key}")
                all_docs.extend(loaded_docs)
                doc_count += 1

        except Exception as e:
            print(f"⚠️ Failed to load {key}: {e}")
            continue

    if not all_docs:
        print("❌ No documents could be loaded.")
        return 0, 0

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"🔬 Created {len(chunks)} chunks from {doc_count} documents")

    # Embed
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save locally to the correct location
    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")
    print("💾 Saved vectorstore locally to faiss_index/")

    # Upload to S3
    try:
        s3.upload_file("faiss_index/index.faiss", index_bucket, "index.faiss")
        s3.upload_file("faiss_index/index.pkl", index_bucket, "index.pkl")
        print("☁️ Uploaded new index to S3")
    except Exception as e:
        print(f"⚠️ Could not upload index to S3: {e}")

    return doc_count, len(chunks)

def get_relevant_chunks(query, vectorstore, k=5):
    try:
        chunks = vectorstore.similarity_search(query, k=10)
        return chunks
    except Exception as e:
        print(f"[Vector Search Error] {e}")
        return []
def rebuild_vectorstore_enriched():
    """
    Full rebuild using enriched chunking — same method as the original
    high quality index. Downloads all docs from S3, uses enrich_pdf_chunks
    for PDFs and chunk_docx_with_metadata for DOCX files, saves locally
    and uploads to S3.
    Returns (doc_count, chunk_count)
    """
    import streamlit as st
    from tools.loaders import enrich_pdf_chunks, chunk_docx_with_metadata
    print("🔄 Starting enriched vectorstore rebuild...")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION")
        )
        docs_bucket = get_secret("S3_DOCS_BUCKET")
        index_bucket = get_secret("S3_INDEX_BUCKET")
    except Exception as e:
        print(f"❌ Could not connect to S3: {e}")
        return 0, 0

    response = s3.list_objects_v2(Bucket=docs_bucket)
    if "Contents" not in response:
        print("❌ No documents found in S3.")
        return 0, 0

    all_chunks = []
    doc_count = 0

    for obj in response["Contents"]:
        key = obj["Key"]
        if not key.endswith((".pdf", ".docx")):
            continue

        try:
            suffix = ".pdf" if key.endswith(".pdf") else ".docx"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                s3.download_file(docs_bucket, key, tmp_file.name)
                print(f"⬇️ Downloaded: {key}")

                clean_name = key.replace(".pdf", "").replace(".docx", "")
                clean_name = clean_name.replace(" ", "_").replace("-", "_").lower()

                if key.endswith(".pdf"):
                    chunks = enrich_pdf_chunks(tmp_file.name)
                else:
                    chunks = chunk_docx_with_metadata(tmp_file.name)

                for chunk in chunks:
                    chunk.metadata["source"] = clean_name

                print(f"📄 {len(chunks)} chunks from {key}")
                all_chunks.extend(chunks)
                doc_count += 1

        except Exception as e:
            print(f"⚠️ Failed to process {key}: {e}")
            continue

    if not all_chunks:
        print("❌ No chunks created.")
        return 0, 0

    print(f"🔬 Total: {len(all_chunks)} chunks from {doc_count} documents")

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    os.makedirs("faiss_index", exist_ok=True)
    vectorstore.save_local("faiss_index")
    print("💾 Saved locally to faiss_index/")

    try:
        s3.upload_file("faiss_index/index.faiss", index_bucket, "index.faiss")
        s3.upload_file("faiss_index/index.pkl", index_bucket, "index.pkl")
        print("☁️ Uploaded new index to S3")
    except Exception as e:
        print(f"⚠️ Could not upload to S3: {e}")

    return doc_count, len(all_chunks)
