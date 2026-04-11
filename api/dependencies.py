from functools import lru_cache
from openai import OpenAI
from tools.s3_utils import get_secret
from tools.embeddings import load_faiss_vectorstore
import os

@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=get_secret("OPENAI_API_KEY"))

@lru_cache(maxsize=4)
def get_vectorstore(tenant: str = "demo"):
    os.environ["TENANT_PREFIX"] = tenant
    from config_loader import reset_config
    reset_config()
    vectorstore, bm25_index = load_faiss_vectorstore(
        "index", get_secret("OPENAI_API_KEY")
    )
    return vectorstore, bm25_index
