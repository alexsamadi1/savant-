"""Extract text and structured metadata from PDF/DOCX documents via GPT."""

import io
import json
import os
import sqlite3
import tempfile
from typing import Dict, List, Optional

import pdfplumber
from docx import Document
from openai import OpenAI

from tools.s3_utils import get_secret
from logic.data_loader import download_sqlite_from_s3, upload_sqlite_to_s3


# --- Text extraction ---

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract all text from a DOCX using python-docx."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Route to the correct text extractor based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes)
    if ext in (".txt", ".md", ".csv"):
        return file_bytes.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported document type: {ext}")


# --- GPT structure extraction ---

DEFAULT_FIELDS = [
    "doc_date", "word_count",
    "has_executive_summary", "has_risk_section", "has_milestones",
    "has_deliverables", "has_budget_section", "section_count",
    "tone_score", "specificity_score", "completeness_score",
    "key_topics",
]


def extract_document_structure(
    text: str,
    filename: str,
    problem_statement: str = "",
    custom_schema: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> Dict:
    """Use GPT to extract structured metadata from document text.

    Args:
        text: Raw document text (will be truncated to 4000 chars).
        filename: Original filename.
        problem_statement: Optional business context.
        custom_schema: Optional list of field names to extract instead of defaults.
        model: OpenAI model to use.

    Returns:
        Dict with extracted fields plus 'filename'.
    """
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    fields = custom_schema or DEFAULT_FIELDS
    truncated = text[:4000]

    system_prompt = (
        "You are a document analyst. Extract structured metadata from the provided text.\n"
        "Return a JSON object with exactly these fields:\n"
        f"{json.dumps(fields)}\n\n"
        "Rules:\n"
        "- doc_date: the document's date as a string (YYYY-MM-DD if possible), or null\n"
        "- word_count: integer count of words in the full text\n"
        "- has_* fields: boolean true/false\n"
        "- section_count: integer number of distinct sections\n"
        "- tone_score: integer 1-10 (1=very informal, 10=very formal)\n"
        "- specificity_score: integer 1-10 (1=vague, 10=highly specific)\n"
        "- completeness_score: integer 1-10 (1=bare outline, 10=comprehensive)\n"
        "- key_topics: list of 3-7 topic strings\n"
        "Return ONLY valid JSON. No markdown fences."
    )

    user_content = f"Filename: {filename}\n"
    if problem_statement:
        user_content += f"Business context: {problem_statement}\n"
    user_content += f"\nDocument text (may be truncated):\n{truncated}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    result = json.loads(response.choices[0].message.content)
    result["filename"] = filename
    return result


# --- SQLite storage ---

def store_doc_metadata_in_sqlite(
    metadata_rows: List[Dict], tenant: str
) -> None:
    """Append document metadata rows to the 'documents' table in the tenant's SQLite DB.

    Downloads the existing DB from S3 (or creates a new one), inserts rows,
    and re-uploads. List/dict values are serialized to JSON strings.
    """
    db_path = download_sqlite_from_s3(tenant)
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Flatten list/dict columns to JSON strings
        cleaned_rows: List[Dict] = []
        for row in metadata_rows:
            cleaned: Dict = {}
            for k, v in row.items():
                if isinstance(v, (list, dict)):
                    cleaned[k] = json.dumps(v)
                else:
                    cleaned[k] = v
            cleaned_rows.append(cleaned)

        if not cleaned_rows:
            conn.close()
            return

        # Build table from first row's keys
        columns = list(cleaned_rows[0].keys())
        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
        cur.execute(f'CREATE TABLE IF NOT EXISTS documents ({col_defs})')

        placeholders = ", ".join("?" for _ in columns)
        col_names = ", ".join(f'"{c}"' for c in columns)
        for row in cleaned_rows:
            values = [row.get(c) for c in columns]
            cur.execute(
                f"INSERT INTO documents ({col_names}) VALUES ({placeholders})",
                values,
            )

        conn.commit()
        conn.close()

        upload_sqlite_to_s3(db_path, tenant)
    finally:
        os.unlink(db_path)


# --- Full pipeline ---

def process_document_full(
    file_bytes: bytes,
    filename: str,
    tenant: str,
    problem_statement: str = "",
    custom_schema: Optional[List[str]] = None,
) -> Dict:
    """Full document processing: extract text → extract structure → store in SQLite.

    Returns:
        The extracted metadata dict.
    """
    text = extract_text(file_bytes, filename)
    metadata = extract_document_structure(
        text, filename, problem_statement, custom_schema
    )
    store_doc_metadata_in_sqlite([metadata], tenant)
    return metadata
