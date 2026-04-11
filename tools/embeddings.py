import os
import json
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
    from tools.s3_utils import get_tenant_prefix
    tenant_prefix = get_tenant_prefix()
    print(f"[TENANT] Loading index for tenant: {tenant_prefix}")

    path = Path(f"faiss_index/{tenant_prefix}")
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
        s3.download_file(bucket, f"{tenant_prefix}/index.faiss", str(faiss_file))
        s3.download_file(bucket, f"{tenant_prefix}/index.pkl", str(pkl_file))
        bm25_file = path / "bm25_index.pkl"
        try:
            s3.download_file(bucket, f"{tenant_prefix}/bm25_index.pkl", str(bm25_file))
        except Exception:
            pass  # BM25 index may not exist yet (pre-hybrid rebuild)
        manifest_file = path / "manifest.json"
        try:
            s3.download_file(bucket, f"{tenant_prefix}/manifest.json", str(manifest_file))
        except Exception:
            pass  # Manifest may not exist yet (pre-pinning rebuild)
        print("✅ Successfully loaded FAISS index from S3")

    except Exception as e:
        print(f"⚠️ Failed to load from S3: {e}")
        if not faiss_file.exists() or not pkl_file.exists():
            print(f"[TENANT] No index found — starting background rebuild...")
            import threading
            def background_rebuild():
                try:
                    from tools.vectorstore_builder import rebuild_vectorstore_enriched
                    doc_count, chunk_count = rebuild_vectorstore_enriched()
                    print(f"[TENANT] Background rebuild complete: {doc_count} docs, {chunk_count} chunks")
                except Exception as rebuild_error:
                    print(f"[TENANT] Background rebuild failed: {rebuild_error}")
            thread = threading.Thread(target=background_rebuild, daemon=True)
            thread.start()
            raise FileNotFoundError(
                f"Knowledge base is being built for the first time. "
                f"Please refresh in 2-3 minutes."
            )

    # Check manifest for embedding model compatibility
    manifest_file = path / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        if manifest.get("embedding_model") != "text-embedding-ada-002":
            print(f"⚠️ WARNING: Index was built with {manifest.get('embedding_model')} but current model is text-embedding-ada-002. Rebuild required.")
        created = manifest.get("created_at", "")
        if created:
            try:
                from datetime import datetime
                created_dt = datetime.fromisoformat(created)
                age_days = (datetime.now() - created_dt.replace(tzinfo=None)).days
                if age_days > 30:
                    print(f"[INDEX] WARNING: Index is {age_days} days old — "
                          f"consider rebuilding if new documents have been added")
            except Exception:
                pass

    # Store index timestamp in environment for log_utils to read
    if manifest_file.exists():
        import os
        try:
            with open(manifest_file) as f:
                m = json.load(f)
            os.environ["SAVANT_INDEX_CREATED_AT"] = m.get(
                "created_at", "")
        except Exception:
            pass

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

