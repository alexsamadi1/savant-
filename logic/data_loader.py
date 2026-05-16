"""Data loading: CSV/Excel → pandas → SQLite → S3, plus query execution."""

import io
import os
import sqlite3
import tempfile
from typing import Dict, List, Optional

import pandas as pd

from tools.s3_utils import get_s3_client, get_secret


def load_csv_to_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel bytes into a pandas DataFrame."""
    ext = os.path.splitext(filename)[1].lower()
    buf = io.BytesIO(file_bytes)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(buf, engine="openpyxl")
    return pd.read_csv(buf)


def dataframe_to_sqlite(df: pd.DataFrame, table_name: str, db_path: str) -> None:
    """Write a DataFrame to a SQLite table (append if DB exists)."""
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def get_schema_description(df: pd.DataFrame, table_name: str) -> Dict:
    """Return schema metadata: row count, columns, dtypes, nulls, stats, samples."""
    columns = []
    for col in df.columns:
        info: Dict = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "sample_values": [
                str(v) for v in df[col].dropna().head(3).tolist()
            ],
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(df[col].min()) if not df[col].isna().all() else None
            info["max"] = float(df[col].max()) if not df[col].isna().all() else None
            info["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
        columns.append(info)
    return {
        "table_name": table_name,
        "row_count": len(df),
        "columns": columns,
    }


def upload_sqlite_to_s3(db_path: str, tenant: str) -> None:
    """Upload a local SQLite file to S3 at {tenant}/data.db."""
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")
    s3.upload_file(db_path, bucket, f"{tenant}/data.db")


def download_sqlite_from_s3(tenant: str) -> Optional[str]:
    """Download tenant's SQLite DB to a temp file. Returns path or None."""
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        s3.download_file(bucket, f"{tenant}/data.db", tmp.name)
        return tmp.name
    except s3.exceptions.NoSuchKey:
        os.unlink(tmp.name)
        return None
    except Exception as e:
        if "404" in str(e) or "NoSuchKey" in str(e) or "Not Found" in str(e):
            os.unlink(tmp.name)
            return None
        raise


def execute_sql_on_tenant_data(sql: str, tenant: str) -> Dict:
    """Download tenant DB, run a SELECT query, return {columns, rows, row_count} or {error}."""
    db_path = download_sqlite_from_s3(tenant)
    if db_path is None:
        return {"error": f"No database found for tenant '{tenant}'"}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        conn.close()
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.unlink(db_path)
