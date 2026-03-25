import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from tools.loaders import enrich_pdf_chunks, chunk_docx_with_metadata

# --- Load API Key ---
def get_openai_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("❌ OPENAI_API_KEY is not set. Please check your .env file or Streamlit secrets.")
    return key

# --- Load Vectorstore ---
def load_faiss_vectorstore(index_name, openai_api_key, index_dir="faiss_index"):
    import boto3, botocore
    from pathlib import Path
    from tools.s3_utils import get_secret

    path = Path(index_dir)
    faiss_file = path / "index.faiss"
    pkl_file = path / "index.pkl"

    path.mkdir(parents=True, exist_ok=True)

    # Try loading from S3 first
    try:
        print("☁️ Attempting to load FAISS index from S3...")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION")
        )
        bucket = get_secret("S3_INDEX_BUCKET")
        s3.download_file(bucket, "index.faiss", str(faiss_file))
        s3.download_file(bucket, "index.pkl", str(pkl_file))
        bm25_file = path / "bm25_index.pkl"
        try:
            s3.download_file(bucket, "bm25_index.pkl", str(bm25_file))
        except Exception:
            pass  # BM25 index may not exist yet (pre-hybrid rebuild)
        print("✅ Successfully loaded FAISS index from S3")

    except botocore.exceptions.BotoCoreError as e:
        print("⚠️ Failed to load from S3, falling back to local. Error:", e)
        if not faiss_file.exists() or not pkl_file.exists():
            raise FileNotFoundError("❌ No local index found either. Cannot load vectorstore.")

    # Load FAISS from local files
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    faiss_vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    # Load BM25 index if present
    bm25_file = path / "bm25_index.pkl"
    bm25_index = None
    if bm25_file.exists():
        with open(bm25_file, "rb") as f:
            bm25_index = pickle.load(f)
        print("✅ BM25 index loaded")
    else:
        print("⚠️ No BM25 index found — falling back to FAISS-only search")

    return faiss_vectorstore, bm25_index

# --- Build and Save Combined Vectorstore ---
def build_combined_vectorstore(pdf_path: str, docx_path: str, index_path: str, api_key: str):
    import toml

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

    # ✅ Upload to S3 after saving locally
    secrets = toml.load(".streamlit/secrets.toml")
    upload_index_to_s3(index_path, secrets["S3_INDEX_BUCKET"])

    return vectorstore

def upload_index_to_s3(index_path: str, bucket: str, secrets_path=".streamlit/secrets.toml"):
    import boto3, toml
    from pathlib import Path

    secrets = toml.load(secrets_path)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=secrets["AWS_REGION"]
    )

    index_files = ["index.faiss", "index.pkl"]
    for file_name in index_files:
        local_path = Path(index_path) / file_name
        s3.upload_file(str(local_path), bucket, file_name)
        print(f"☁️ Uploaded {file_name} to S3 bucket {bucket}")

