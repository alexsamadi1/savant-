from fastapi import APIRouter
from api.models import HealthResponse
from tools.s3_utils import get_s3_client, get_secret

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(tenant: str = "demo"):
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")

    has_data = False
    has_analysis = False
    last_updated = None

    try:
        s3.head_object(Bucket=bucket, Key=f"{tenant}/data.db")
        has_data = True
    except Exception:
        pass

    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{tenant}/analysis_status.json")
        import json
        status_data = json.loads(resp["Body"].read())
        has_analysis = status_data.get("status") == "complete"
        last_updated = resp["LastModified"].strftime("%b %d, %Y")
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        tenant=tenant,
        has_data=has_data,
        has_analysis=has_analysis,
        last_updated=last_updated,
    )
