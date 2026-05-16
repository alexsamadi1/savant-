"""Dashboard route: serve cached analysis results."""

import json

from fastapi import APIRouter, HTTPException

from api.models import DashboardConfig
from tools.s3_utils import get_s3_client, get_secret

router = APIRouter()


@router.get("/dashboard/{tenant}", response_model=DashboardConfig)
async def get_dashboard(tenant: str):
    """Return the cached DashboardConfig for a tenant."""
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{tenant}/dashboard.json")
        data = json.loads(resp["Body"].read())
        return DashboardConfig(**data)
    except Exception:
        raise HTTPException(status_code=404, detail=f"No dashboard found for tenant '{tenant}'")
