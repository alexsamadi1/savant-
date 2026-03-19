import csv
from datetime import datetime
import os
from tools.s3_utils import download_file_from_s3, upload_file_to_s3, get_secret

LOG_FILE = "query_logs.csv"
S3_KEY = f"logs/{LOG_FILE}"

def ensure_log_file_exists():
    if not os.path.exists(LOG_FILE):
        try:
            s3_bucket = get_secret("S3_DOCS_BUCKET")
            download_file_from_s3(S3_KEY, s3_bucket)
            print(f"[LOG] Pulled {LOG_FILE} from S3")
        except Exception as e:
            print(f"[LOG] No existing log on S3 or error downloading: {e}")
        else:
            return

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "session_id", "question", "response",
                "fallback", "response_type", "user_role", "user_tenure", "source_docs", "feedback"
            ])

def log_query_to_csv(
    question: str,
    response: str,
    fallback: bool = False,
    response_type: str = "direct",
    user_role: str = None,
    user_tenure: str = None,
    source_docs: list = None,
    feedback: str = ""
):
    import streamlit as st
    ensure_log_file_exists()
    s3_bucket = get_secret("S3_DOCS_BUCKET")

    session_id = st.session_state.get("session_id")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id

    row = [
        datetime.now().isoformat(),
        session_id,
        question.strip(),
        response.strip(),
        fallback,
        response_type,
        user_role or "",
        user_tenure or "",
        ", ".join(source_docs) if source_docs else "",
        feedback
    ]

    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        with open(LOG_FILE, "rb") as f:
            upload_file_to_s3(f, S3_KEY, s3_bucket)
        print("[LOG] Uploaded updated log to S3.")
    except Exception as e:
        print(f"[LOG] Logging failed: {e}")

def log_chat_interaction(
    user_input,
    answer,
    user_profile,
    source_docs,
    fallback=False,
    response_type="direct",
    feedback=""
):
    log_query_to_csv(
        question=user_input,
        response=answer,
        fallback=fallback,
        response_type=response_type,
        user_role=user_profile.get("role", "Unknown"),
        user_tenure=user_profile.get("tenure", "Unknown"),
        source_docs=source_docs,
        feedback=feedback
    )
