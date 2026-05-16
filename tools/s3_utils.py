import boto3
import os
from pathlib import Path

_secrets_cache: dict = {}


def get_secret(key: str) -> str:
    """Load a secret from secrets.toml (cached) or environment variables."""
    # Load toml once and cache
    if not _secrets_cache:
        import toml as _toml
        for candidate in ("secrets.toml", ".streamlit/secrets.toml"):
            p = Path(__file__).resolve().parent.parent / candidate
            if p.is_file():
                _secrets_cache.update(_toml.load(str(p)))
                break

    # Flat lookup
    if key in _secrets_cache:
        return str(_secrets_cache[key])

    # Nested lookup — search all sub-dicts
    for v in _secrets_cache.values():
        if isinstance(v, dict) and key in v:
            return str(v[key])

    # Fall back to environment
    env_val = os.environ.get(key)
    if env_val:
        return env_val

    raise ValueError(f"Secret '{key}' not found in secrets.toml or environment")

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
        region_name=get_secret("AWS_REGION")
    )

def get_tenant_prefix() -> str:
    import os
    # Shell env var takes priority over secrets.toml
    # This allows: TENANT_PREFIX=innovim streamlit run app.py
    env_val = os.environ.get("TENANT_PREFIX", "").strip("/")
    if env_val:
        return env_val
    try:
        val = get_secret("TENANT_PREFIX").strip("/")
        return val if val else "demo"
    except Exception:
        return "demo"

def upload_file_to_s3(file, filename, bucket):
    s3 = get_s3_client()
    s3.upload_fileobj(file, bucket, filename)

def list_files_in_bucket(bucket_name):
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket_name)
    return [item["Key"] for item in response.get("Contents", [])]

def download_s3_file_to_tmp(bucket, key):
    s3 = get_s3_client()
    local_path = f"/tmp/{key.replace('/', '_')}"
    s3.download_file(bucket, key, local_path)
    return local_path

def download_faiss_index_from_s3(local_path="/tmp/faiss_index"):
    s3 = get_s3_client()
    Path(local_path).mkdir(parents=True, exist_ok=True)
    bucket = get_secret("S3_INDEX_BUCKET")
    prefix = "faiss_index/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = key.split("/")[-1]
        dest_path = os.path.join(local_path, filename)
        s3.download_file(bucket, key, dest_path)
    return local_path

def download_file_from_s3(s3_key, bucket_name, local_path=None):
    s3 = get_s3_client()
    if local_path is None:
        local_path = s3_key.split("/")[-1]
    s3.download_file(bucket_name, s3_key, local_path)
    return local_path
