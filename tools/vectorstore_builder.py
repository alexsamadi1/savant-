import os
import re
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import boto3
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tools.s3_utils import get_secret
from datetime import datetime
import nltk
from nltk.stem import PorterStemmer
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# --- File Map ---
# tokenize_for_bm25()          Line ~30  — BM25 tokenizer (stemming, stopwords)
# get_relevant_chunks()        Line ~180 — Hybrid retrieval (FAISS + BM25 + RRF)
# rebuild_vectorstore_from_s3() Line ~80 — Basic rebuild from S3
# rebuild_vectorstore_enriched() Line ~300 — Full rebuild with contextual embeddings
# add_contextual_embeddings()  Line ~240 — GPT-powered chunk contextualization
# _save_and_upload_index()     Line ~270 — Save locally + upload to S3

_stemmer = PorterStemmer()

def tokenize_for_bm25(text: str) -> list:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    raw_tokens = text.split()
    result = []
    for raw in raw_tokens:
        lower = raw.lower()
        if lower in stop_words:
            continue
        if re.search(r'\d', raw) or (raw.isupper() and len(raw) <= 10):
            result.append(lower)
        else:
            result.append(_stemmer.stem(lower))
    return result

# --- Load API Key ---
def get_openai_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("❌ OPENAI_API_KEY is not set. Please check your .env file or Streamlit secrets.")
    return key

def rebuild_vectorstore_from_s3():
    """
    Full rebuild — downloads all docs from S3, embeds them,
    saves locally, and uploads new index back to S3.
    Returns (doc_count, chunk_count)
    """
    import streamlit as st
    print("🔄 Starting full vectorstore rebuild from S3...")

    from tools.s3_utils import get_tenant_prefix
    tenant_prefix = get_tenant_prefix()
    print(f"[TENANT] Rebuilding index for tenant: {tenant_prefix}")

    try:
        from botocore.config import Config as BotoConfig
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION"),
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 2}
            )
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
        if not key.startswith(f"{tenant_prefix}/"):
            continue
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
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"🔬 Created {len(chunks)} chunks from {doc_count} documents")

    # Embed
    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(chunks, embeddings)

    _save_and_upload_index(vectorstore, chunks, s3, index_bucket, tenant_prefix)

    return doc_count, len(chunks)

def get_relevant_chunks(query, vectorstore, k=20, bm25_index=None):
    try:
        faiss_results = vectorstore.similarity_search(query, k=k)

        if bm25_index is None:
            return faiss_results

        from langchain_core.documents import Document as LCDocument
        bm25_obj, bm25_docs, bm25_metas = bm25_index if len(bm25_index) == 3 else (*bm25_index, [{} for _ in bm25_index[1]])

        tokenized_query = tokenize_for_bm25(query)
        scores = bm25_obj.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        bm25_results = [LCDocument(page_content=bm25_docs[i], metadata=bm25_metas[i]) for i in top_indices]

        # Reciprocal Rank Fusion
        rrf_scores = {}
        doc_map = {}

        for rank, doc in enumerate(faiss_results):
            key = hashlib.md5((doc.page_content + doc.metadata.get('source', '')).encode()).hexdigest()
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_results):
            key = hashlib.md5((doc.page_content + doc.metadata.get('source', '')).encode()).hexdigest()
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + 60)
            doc_map[key] = doc

        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[key] for key in sorted_keys[:k]]

    except Exception as e:
        print(f"[Vector Search Error] {e}")
        return []
def add_contextual_embeddings(chunks, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    print(f"🧠 Adding contextual embeddings to {len(chunks)} chunks...")

    def contextualize(chunk):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document indexing assistant. Your job is to situate this chunk within its source document. "
                            "Write exactly 2 sentences. Sentence 1: describe what document and specific section this chunk comes from. "
                            "Sentence 2: describe what specific information, policy, or procedure this chunk contains."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Document source: {chunk.metadata.get('source', 'unknown')}\n"
                            f"Section: {chunk.metadata.get('section_title', 'unknown')}\n"
                            f"Chunk content: {chunk.page_content[:500]}\n\n"
                            "Write exactly 2 sentences as instructed."
                        )
                    }
                ],
                temperature=0
            )
            context_sentence = response.choices[0].message.content.strip()
            chunk.metadata["context_summary"] = context_sentence
        except Exception:
            pass  # leave chunk unchanged on failure
        return chunk

    with ThreadPoolExecutor(max_workers=5) as executor:
        chunks = list(executor.map(contextualize, chunks))

    return chunks


def _save_and_upload_index(vectorstore, all_chunks, s3, index_bucket, tenant_prefix):
    index_dir = f"faiss_index/{tenant_prefix}"
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"💾 Saved locally to {index_dir}/")

    from rank_bm25 import BM25Okapi
    texts = [chunk.page_content for chunk in all_chunks]
    metadatas = [chunk.metadata for chunk in all_chunks]
    bm25_corpus = [tokenize_for_bm25(t) for t in texts]
    bm25_obj = BM25Okapi(bm25_corpus)
    with open(f"{index_dir}/bm25_index.pkl", "wb") as f:
        pickle.dump((bm25_obj, texts, metadatas), f)

    manifest = {
        "embedding_model": "text-embedding-ada-002",
        "created_at": datetime.now().isoformat()
    }
    with open(f"{index_dir}/manifest.json", "w") as f:
        json.dump(manifest, f)

    try:
        s3.upload_file(f"{index_dir}/index.faiss", index_bucket, f"{tenant_prefix}/index.faiss")
        s3.upload_file(f"{index_dir}/index.pkl", index_bucket, f"{tenant_prefix}/index.pkl")
        s3.upload_file(f"{index_dir}/bm25_index.pkl", index_bucket, f"{tenant_prefix}/bm25_index.pkl")
        s3.upload_file(f"{index_dir}/manifest.json", index_bucket, f"{tenant_prefix}/manifest.json")
        print("☁️ Uploaded new index to S3")
    except Exception as e:
        print(f"⚠️ Could not upload to S3: {e}")

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

    from tools.s3_utils import get_tenant_prefix
    tenant_prefix = get_tenant_prefix()
    print(f"[TENANT] Rebuilding index for tenant: {tenant_prefix}")

    try:
        from botocore.config import Config as BotoConfig
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION"),
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 2}
            )
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
        if not key.startswith(f"{tenant_prefix}/"):
            continue
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

    all_chunks = add_contextual_embeddings(all_chunks, get_openai_api_key())

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(all_chunks, embeddings)

    _save_and_upload_index(vectorstore, all_chunks, s3, index_bucket, tenant_prefix)

    return doc_count, len(all_chunks)
