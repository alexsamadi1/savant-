"""Document ingestion: text chunking, FAISS index build/load, and S3 persistence."""

import os
import shutil
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from tools.s3_utils import get_s3_client, get_secret
from logic.doc_structure_extractor import extract_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def build_faiss_index(chunks: List[str], tenant: str, metadatas: Optional[List[dict]] = None) -> None:
    """Embed chunks, build a FAISS index, upload to S3, and clean up local files."""
    embeddings = OpenAIEmbeddings(api_key=get_secret("OPENAI_API_KEY"))
    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)

    local_dir = f"/tmp/faiss_{tenant}"
    vectorstore.save_local(local_dir)

    # Upload all index files to S3
    s3 = get_s3_client()
    bucket = get_secret("S3_INDEX_BUCKET")
    for fname in os.listdir(local_dir):
        local_path = os.path.join(local_dir, fname)
        if os.path.isfile(local_path):
            s3.upload_file(local_path, bucket, f"{tenant}/faiss/{fname}")

    shutil.rmtree(local_dir)


def load_faiss_index(tenant: str) -> Optional[FAISS]:
    """Download FAISS index from S3 and load it. Returns None if not found."""
    s3 = get_s3_client()
    bucket = get_secret("S3_INDEX_BUCKET")
    prefix = f"{tenant}/faiss/"

    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    except Exception:
        return None

    contents = response.get("Contents", [])
    if not contents:
        return None

    local_dir = f"/tmp/faiss_{tenant}"
    os.makedirs(local_dir, exist_ok=True)

    for obj in contents:
        key = obj["Key"]
        fname = key.split("/")[-1]
        if fname:
            s3.download_file(bucket, key, os.path.join(local_dir, fname))

    embeddings = OpenAIEmbeddings(api_key=get_secret("OPENAI_API_KEY"))
    vectorstore = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)

    shutil.rmtree(local_dir)
    return vectorstore


def ingest_document_to_faiss(file_bytes: bytes, filename: str, tenant: str) -> None:
    """Extract text from a document, chunk it, and build/update a FAISS index."""
    text = extract_text(file_bytes, filename)
    chunks = chunk_text(text)
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
    build_faiss_index(chunks, tenant, metadatas=metadatas)
