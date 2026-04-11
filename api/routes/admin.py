import os
from fastapi import APIRouter, Header, HTTPException
from tools.s3_utils import get_secret
from api.dependencies import get_vectorstore
from tools.gap_analysis import run_gap_analysis, run_conflict_detection
import boto3

router = APIRouter()


@router.post("/admin/gap-analysis")
def gap_analysis(
    body: dict = {},
    x_admin_code: str = Header(..., alias="X-Admin-Code"),
):
    tenant = body.get("tenant", "demo")
    if x_admin_code != get_secret("ADMIN_CODE"):
        raise HTTPException(status_code=403, detail="Invalid admin code")

    os.environ["TENANT_PREFIX"] = tenant
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
        region_name=get_secret("AWS_REGION"),
    )
    result = run_gap_analysis(s3_client, get_secret("S3_DOCS_BUCKET"))
    return result


@router.post("/admin/conflicts")
def conflicts(
    body: dict = {},
    x_admin_code: str = Header(..., alias="X-Admin-Code"),
):
    tenant = body.get("tenant", "demo")
    if x_admin_code != get_secret("ADMIN_CODE"):
        raise HTTPException(status_code=403, detail="Invalid admin code")

    vectorstore, bm25_index = get_vectorstore(tenant)
    result = run_conflict_detection(
        vectorstore, bm25_index, get_secret("OPENAI_API_KEY")
    )
    return result
