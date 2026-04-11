from fastapi import APIRouter
from api.models import HealthResponse
from api.dependencies import get_vectorstore
from tools.s3_utils import get_secret
import boto3

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health(tenant: str = "demo"):
    try:
        vectorstore, _ = get_vectorstore(tenant)
        loaded = vectorstore is not None
    except Exception:
        loaded = False

    doc_count = 0
    last_updated = None
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION")
        )
        resp = s3.list_objects_v2(
            Bucket=get_secret("S3_DOCS_BUCKET"),
            Prefix=f"{tenant}/"
        )
        latest = None
        for o in resp.get("Contents", []):
            if o["Key"].endswith((".pdf", ".docx")):
                doc_count += 1
                if latest is None or o["LastModified"] > latest:
                    latest = o["LastModified"]
        if latest:
            last_updated = latest.strftime("%b %-d, %Y")
    except Exception:
        pass

    return HealthResponse(
        status="ok" if loaded else "degraded",
        vectorstore_loaded=loaded,
        tenant=tenant,
        doc_count=doc_count,
        last_updated=last_updated,
    )
