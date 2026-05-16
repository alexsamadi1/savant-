"""Ingest routes: upload CSV/Excel data and PDF/DOCX documents."""

import json
from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from api.models import IngestResponse
from logic.data_loader import (
    dataframe_to_sqlite,
    download_sqlite_from_s3,
    get_schema_description,
    load_csv_to_dataframe,
    upload_sqlite_to_s3,
)
from logic.doc_ingestor import ingest_document_to_faiss
from logic.doc_structure_extractor import process_document_full
from tools.s3_utils import get_s3_client, get_secret

import os
import tempfile

router = APIRouter(prefix="/ingest")


@router.post("/data", response_model=IngestResponse)
async def ingest_data(
    file: UploadFile = File(...),
    tenant: str = Form(...),
    problem_statement: str = Form(""),
):
    """Upload CSV/Excel → pandas → SQLite → S3. Saves schema.json."""
    file_bytes = await file.read()
    filename = file.filename or "upload.csv"

    # Load into DataFrame
    df = load_csv_to_dataframe(file_bytes, filename)

    # Derive table name from filename
    table_name = os.path.splitext(filename)[0].lower().replace(" ", "_").replace("-", "_")

    # Download existing DB or create new one
    db_path = download_sqlite_from_s3(tenant)
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

    # Write DataFrame to SQLite and upload
    dataframe_to_sqlite(df, table_name, db_path)
    upload_sqlite_to_s3(db_path, tenant)
    os.unlink(db_path)

    # Build and save schema
    schema = get_schema_description(df, table_name)

    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")

    # Load existing schemas or start fresh
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{tenant}/schema.json")
        existing_schemas = json.loads(resp["Body"].read())
    except Exception:
        existing_schemas = []

    # Replace schema for this table or append
    existing_schemas = [s for s in existing_schemas if s.get("table_name") != table_name]
    existing_schemas.append(schema)

    s3.put_object(
        Bucket=bucket,
        Key=f"{tenant}/schema.json",
        Body=json.dumps(existing_schemas, default=str),
        ContentType="application/json",
    )

    return IngestResponse(
        tenant=tenant,
        files_uploaded=[filename],
        schema_detected=schema,
        ready_for_analysis=True,
    )


@router.post("/documents")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    tenant: str = Form(...),
    problem_statement: str = Form(""),
):
    """Upload PDF/DOCX: FAISS indexing + structure extraction in sequence."""
    results = []
    for upload in files:
        file_bytes = await upload.read()
        filename = upload.filename or "document"

        # Pipeline 1: FAISS indexing
        ingest_document_to_faiss(file_bytes, filename, tenant)

        # Pipeline 2: Structure extraction → SQLite
        metadata = process_document_full(
            file_bytes, filename, tenant, problem_statement
        )

        results.append({"filename": filename, "metadata": metadata})

    return results
