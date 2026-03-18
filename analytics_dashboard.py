import streamlit as st
import pandas as pd
import collections
import re
import boto3
from datetime import datetime
from config_loader import get_config

STOPWORDS = {
    "what", "when", "where", "which", "who", "will", "with", "that", "this",
    "have", "from", "they", "been", "were", "would", "could", "should", "about",
    "many", "much", "more", "some", "than", "then", "them", "their", "there",
    "your", "does", "just", "into", "also", "make", "take", "know", "need",
    "like", "time", "work", "days", "info", "update"
}

def clean_source_name(raw):
    name = re.sub(r"_page_\d+", "", raw)
    name = name.replace("_", " ").strip().title()
    return name

def show_analytics_dashboard():
    cfg = get_config()
    brand = cfg["brand"]

    st.title(f"📊 {brand['app_name']} Analytics")

    if not st.session_state.get("is_admin", False):
        st.error("⛔ Access denied.")
        return

    try:
        df = pd.read_csv("query_logs.csv", usecols=[
            "timestamp", "session_id", "question", "response",
            "fallback", "response_type", "user_role", "user_tenure",
            "source_docs", "feedback"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["fallback"] = pd.to_numeric(df["fallback"], errors="coerce").fillna(0)
    except FileNotFoundError:
        st.warning("No query log file found yet.")
        return
    except Exception as e:
        st.error(f"Error loading log data: {e}")
        return

    show_documents_panel()
    st.markdown("---")
    show_usage_summary(df)
    show_answer_quality(df)
    show_unanswered_questions(df)
    show_recent_activity(df)
    show_top_questions(df)
    show_top_keywords(df)
    show_user_demographics(df)
    show_bot_performance(df)
    show_source_documents(df)
    show_sessions(df)

    st.markdown("---")
    if st.button("🔙 Back to Assistant"):
        st.session_state.show_analytics = False
        st.rerun()


def show_documents_panel():
    cfg = get_config()
    st.subheader("📁 Loaded documents")
    st.caption("Documents currently in the knowledge base")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"]
        )
        bucket = st.secrets["S3_DOCS_BUCKET"]
        response = s3.list_objects_v2(Bucket=bucket)

        if "Contents" not in response:
            st.info("No documents uploaded yet.")
            return

        docs = []
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith((".pdf", ".docx")):
                continue
            ext = key.split(".")[-1].upper()
            size_kb = round(obj["Size"] / 1024, 1)
            uploaded = obj["LastModified"].strftime("%b %d, %Y")
            clean_name = key.replace("_", " ").replace("-", " ").title()
            docs.append({
                "Document": clean_name,
                "Type": ext,
                "Size": f"{size_kb} KB",
                "Uploaded": uploaded
            })

        if docs:
            docs_df = pd.DataFrame(docs)
            st.dataframe(docs_df, use_container_width=True, hide_index=True)
            st.caption(f"{len(docs)} document{'s' if len(docs) != 1 else ''} loaded")
        else:
            st.info("No documents found in storage.")

    except Exception as e:
        st.warning(f"Could not load document list: {e}")


def show_usage_summary(df):
    st.subheader("📈 Usage overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total queries", len(df))
    col2.metric("Unique sessions", df["session_id"].nunique() if "session_id" in df.columns else "—")

    if "fallback" in df.columns:
        fallback_rate = df["fallback"].mean() * 100
        col3.metric("Fallback rate", f"{fallback_rate:.1f}%")

    avg_per_session = len(df) / max(df["session_id"].nunique(), 1)
    col4.metric("Avg queries per session", f"{avg_per_session:.1f}")

    daily = df.groupby(df["timestamp"].dt.date).size()
    st.line_chart(daily.rename("Daily queries"))


def show_answer_quality(df):
    st.subheader("✅ Answer quality")

    if "fallback" not in df.columns:
        return

    total = len(df)
    direct = int((df["fallback"] == 0).sum())
    fallback = int((df["fallback"] == 1).sum())
    direct_pct = round((direct / total) * 100, 1) if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Answered directly from documents", f"{direct_pct}%")
    col2.metric("Direct answers", direct)
    col3.metric("Fallback answers", fallback)

    quality_df = pd.DataFrame({
        "Type": ["Answered from documents", "Fallback / unclear"],
        "Count": [direct, fallback]
    })
    st.bar_chart(quality_df.set_index("Type"))


def show_unanswered_questions(df):
    st.subheader("⚠️ Questions that need better documents")
    st.caption("These questions triggered fallback responses — consider uploading documents that cover these topics")

    if "fallback" not in df.columns:
        return

    unanswered = (
        df[df["fallback"] == 1][["timestamp", "question"]]
        .dropna()
        .sort_values("timestamp", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    if unanswered.empty:
        st.success("No fallback questions — the bot is answering everything confidently.")
        return

    unanswered["timestamp"] = unanswered["timestamp"].dt.strftime("%b %d, %Y %I:%M %p")
    unanswered.columns = ["Asked on", "Question"]
    st.dataframe(unanswered, use_container_width=True, hide_index=True)


def show_recent_activity(df):
    st.subheader("🕐 Recent activity")
    st.caption("Last 10 questions asked")

    recent = (
        df[["timestamp", "question", "source_docs", "fallback"]]
        .dropna(subset=["question"])
        .sort_values("timestamp", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    if recent.empty:
        st.info("No activity yet.")
        return

    recent["timestamp"] = recent["timestamp"].dt.strftime("%b %d %I:%M %p")
    recent["source_docs"] = recent["source_docs"].fillna("—").apply(
        lambda x: clean_source_name(x.split(",")[0].strip()) if x != "—" else "—"
    )
    recent["fallback"] = recent["fallback"].apply(
        lambda x: "⚠️ Fallback" if x == 1 else "✅ Direct"
    )
    recent.columns = ["Time", "Question", "Source", "Status"]
    st.dataframe(recent, use_container_width=True, hide_index=True)


def show_top_questions(df):
    st.subheader("📌 Most frequently asked questions")
    if "question" not in df.columns:
        return

    q_counts = (
        df["question"]
        .dropna()
        .str.strip()
        .str.lower()
        .value_counts()
        .head(10)
        .reset_index()
    )
    q_counts.columns = ["Question", "Count"]
    q_counts["Question"] = q_counts["Question"].str.capitalize()
    st.dataframe(q_counts, use_container_width=True, hide_index=True)


def show_top_keywords(df):
    st.subheader("🔍 Top keywords")
    all_words = " ".join(df["question"].fillna("")).lower()
    words = re.findall(r"\b\w{4,}\b", all_words)
    filtered = [w for w in words if w not in STOPWORDS]
    common = collections.Counter(filtered).most_common(10)
    word_df = pd.DataFrame(common, columns=["Keyword", "Count"])
    st.dataframe(word_df, use_container_width=True, hide_index=True)


def show_user_demographics(df):
    st.subheader("👥 User demographics")
    col1, col2 = st.columns(2)

    if "user_role" in df.columns:
        role_counts = df["user_role"].value_counts()
        col1.markdown("**Role**")
        col1.bar_chart(role_counts)

    if "user_tenure" in df.columns:
        tenure_counts = df["user_tenure"].value_counts()
        tenure_counts = tenure_counts[tenure_counts > 0]
        col2.markdown("**Tenure**")
        col2.bar_chart(tenure_counts)


def show_bot_performance(df):
    st.subheader("🤖 Bot performance")

    if "fallback" in df.columns:
        fallback_daily = (
            df.groupby(df["timestamp"].dt.date)["fallback"]
            .mean()
            .mul(100)
            .clip(lower=0)
        )
        st.markdown("**Fallback rate over time**")
        st.line_chart(fallback_daily.rename("Fallback %"))

    if "response_type" in df.columns:
        st.markdown("**Response type breakdown**")
        type_counts = df["response_type"].value_counts()
        st.bar_chart(type_counts)


def show_source_documents(df):
    if "source_docs" not in df.columns:
        return

    st.subheader("📄 Top source documents")
    exploded = df["source_docs"].dropna().str.split(", ")
    flat = exploded.explode().str.strip()
    flat = flat[flat != ""]
    flat_clean = flat.apply(clean_source_name)
    doc_counts = flat_clean.value_counts().head(10)
    st.bar_chart(doc_counts.rename("Mentions"))


def show_sessions(df):
    if "session_id" not in df.columns:
        return

    st.subheader("🧭 Session analytics")
    col1, col2 = st.columns(2)
    col1.metric("Unique sessions", df["session_id"].nunique())
    avg_per_session = len(df) / max(df["session_id"].nunique(), 1)
    col2.metric("Avg queries per session", f"{avg_per_session:.1f}")