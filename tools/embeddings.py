import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tools.loaders import enrich_pdf_chunks, chunk_docx_with_metadata
from tools.s3_utils import get_secret, get_s3_client

def load_faiss_vectorstore(index_name, openai_api_key, index_dir="faiss_index"):
    import botocore
    path = Path(index_dir)
    faiss_file = path / "index.faiss"
    pkl_file = path / "index.pkl"
    path.mkdir(parents=True, exist_ok=True)

    try:
        print("☁️ Attempting to load FAISS index from S3...")
        s3 = get_s3_client()
        bucket = get_secret("S3_INDEX_BUCKET")
        s3.download_file(bucket, "index.faiss", str(faiss_file))
        s3.download_file(bucket, "index.pkl", str(pkl_file))
        print("✅ Successfully loaded FAISS index from S3")
    except Exception as e:
        print("⚠️ Failed to load from S3, falling back to local. Error:", e)
        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError("❌ No local index found either. Cannot load vectorstore.")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def build_combined_vectorstore(pdf_path: str, docx_path: str, index_path: str, api_key: str):
    print("📥 Enriching PDF handbook...")
    pdf_chunks = enrich_pdf_chunks(pdf_path)

    print("📥 Chunking DOCX orientation guide...")
    docx_chunks = chunk_docx_with_metadata(docx_path)

    all_chunks = pdf_chunks + docx_chunks
    print(f"✅ Total chunks: {len(all_chunks)}")

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(index_path)
    print(f"✅ Vectorstore saved to: {index_path}/")

    upload_index_to_s3(index_path, get_secret("S3_INDEX_BUCKET"))
    return vectorstore

def upload_index_to_s3(index_path: str, bucket: str):
    s3 = get_s3_client()
    index_files = ["index.faiss", "index.pkl"]
    for file_name in index_files:
        local_path = Path(index_path) / file_name
        s3.upload_file(str(local_path), bucket, file_name)
        print(f"☁️ Uploaded {file_name} to S3 bucket {bucket}")
