import os
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException
from api.models import UploadResponse, RebuildResponse
from tools.s3_utils import get_secret, upload_file_to_s3
from tools.vectorstore_builder import rebuild_vectorstore_enriched
import boto3

router = APIRouter()


@router.get("/documents")
def list_documents(tenant: str = "demo"):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
        region_name=get_secret("AWS_REGION"),
    )
    resp = s3.list_objects_v2(
        Bucket=get_secret("S3_DOCS_BUCKET"),
        Prefix=f"{tenant}/",
    )
    docs = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if not key.endswith((".pdf", ".docx")):
            continue
        name = key.split("/", 1)[-1] if "/" in key else key
        doc_type = "PDF" if key.endswith(".pdf") else "DOCX"
        size_kb = round(obj["Size"] / 1024, 1)
        uploaded = obj["LastModified"].strftime("%b %d, %Y")
        docs.append({
            "name": name.replace("_", " ").strip().title(),
            "type": doc_type,
            "size_kb": size_kb,
            "uploaded": uploaded,
        })
    return docs


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tenant: str = Form("demo"),
    x_admin_code: str = Header(..., alias="X-Admin-Code"),
):
    if x_admin_code != get_secret("ADMIN_CODE"):
        raise HTTPException(status_code=403, detail="Invalid admin code")

    filename = file.filename
    if filename and filename.endswith(".docx"):
        from tools.filename_generator import generate_smart_filename
        content = await file.read()
        await file.seek(0)
        smart_name = generate_smart_filename(content.decode("utf-8", errors="ignore"), filename)
        filename = smart_name

    os.environ["TENANT_PREFIX"] = tenant
    bucket = get_secret("S3_DOCS_BUCKET")
    s3_key = f"{tenant}/{filename}"
    upload_file_to_s3(file.file, s3_key, bucket)

    doc_count, chunk_count = rebuild_vectorstore_enriched()

    return UploadResponse(
        filename=filename or "unknown",
        doc_count=doc_count,
        chunk_count=chunk_count,
    )


@router.post("/rebuild", response_model=RebuildResponse)
def rebuild(
    body: dict = {},
    x_admin_code: str = Header(..., alias="X-Admin-Code"),
):
    tenant = body.get("tenant", "demo")
    if x_admin_code != get_secret("ADMIN_CODE"):
        raise HTTPException(status_code=403, detail="Invalid admin code")

    os.environ["TENANT_PREFIX"] = tenant
    doc_count, chunk_count = rebuild_vectorstore_enriched()

    return RebuildResponse(doc_count=doc_count, chunk_count=chunk_count)
