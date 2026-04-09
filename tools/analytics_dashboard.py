from tools.s3_utils import get_secret
import streamlit as st
try:
    from tools.s3_utils import get_secret
except ImportError:
    pass
import pandas as pd
import json
import collections
import re
import boto3
from datetime import datetime
from openai import OpenAI
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

def _ensure_gap_analysis():
    """Auto-run gap analysis on first dashboard load, cache in session_state."""
    if "gap_analysis_result" in st.session_state:
        return
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION"),
        )
        bucket = get_secret("S3_DOCS_BUCKET")
        with st.spinner("Running initial knowledge base health check..."):
            from tools.gap_analysis import run_gap_analysis
            st.session_state["gap_analysis_result"] = run_gap_analysis(s3, bucket)
    except Exception as e:
        print(f"[GAP ANALYSIS] Auto-run failed: {e}")
        st.session_state["gap_analysis_result"] = None


def _show_health_score():
    """Render the health score metric with color coding."""
    result = st.session_state.get("gap_analysis_result")
    if not result:
        st.warning("Health score unavailable — gap analysis did not complete.")
        return
    score = result.get("health_score", 0)
    explanation = result.get("health_explanation", "")
    if score >= 70:
        color = "#00C9A7"
    elif score >= 40:
        color = "#f0ad4e"
    else:
        color = "#d9534f"
    st.markdown(
        f"<div style='text-align:center;margin-bottom:1rem;'>"
        f"<span style='font-size:3rem;font-weight:700;color:{color};'>{score}</span>"
        f"<span style='font-size:1.2rem;color:#888;'>/100</span>"
        f"<p style='color:#aaa;font-size:0.95rem;margin-top:0.25rem;'>{explanation}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def show_analytics_dashboard():
    from tools.log_utils import LOG_FILE

    cfg = get_config()
    brand = cfg["brand"]

    st.title(f"📊 {brand['app_name']} Analytics")

    if not st.session_state.get("is_admin", False):
        st.error("⛔ Access denied.")
        return

    try:
        all_cols = pd.read_csv(LOG_FILE, nrows=0).columns.tolist()
        use_cols = [
            "timestamp", "session_id", "question", "response",
            "fallback", "response_type", "user_role", "user_tenure",
            "source_docs", "feedback"
        ]
        if "gap_reason" in all_cols:
            use_cols.append("gap_reason")
        df = pd.read_csv(LOG_FILE, usecols=use_cols)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["fallback"] = pd.to_numeric(df["fallback"], errors="coerce").fillna(0)
    except FileNotFoundError:
        st.warning("No query log file found yet.")
        return
    except Exception as e:
        st.error(f"Error loading log data: {e}")
        return

    # Auto-run gap analysis for health score
    _ensure_gap_analysis()

    tab_overview, tab_gaps, tab_conflicts, tab_usage, tab_docs = st.tabs([
        "Overview", "Knowledge Gaps", "Conflicts", "Usage & Activity", "Documents"
    ])

    with tab_overview:
        _show_health_score()
        show_usage_summary(df)
        show_answer_quality(df)

    with tab_gaps:
        show_gap_analysis_panel()
        st.markdown("---")
        show_unanswered_questions(df)

    with tab_conflicts:
        show_conflict_detection_panel()

    with tab_usage:
        show_recent_activity(df)
        show_top_questions(df)
        show_top_keywords(df)
        show_user_demographics(df)
        show_bot_performance(df)
        show_sessions(df)

    with tab_docs:
        show_documents_panel()
        st.markdown("---")
        show_source_documents(df)
        # Stale & underperforming from cached gap analysis
        gap_result = st.session_state.get("gap_analysis_result")
        if gap_result:
            stale = gap_result.get("stale_docs", [])
            if stale:
                st.markdown("---")
                st.subheader("🕰️ Stale Documents")
                st.caption("Documents uploaded more than 180 days ago that may need review")
                for doc in stale:
                    filename = doc.get("filename", "")
                    days = doc.get("days_since_upload", 0)
                    rec = doc.get("recommendation", "")
                    clean = filename.replace("_", " ").replace("-", " ").title()
                    st.markdown(f"- **{clean}** — {days} days old. {rec}")
            underperforming = gap_result.get("underperforming_docs", [])
            if underperforming:
                st.markdown("---")
                st.subheader("📉 Underperforming Documents")
                st.caption("Documents that exist but are rarely or never cited in answers")
                for doc in underperforming:
                    filename = doc.get("filename", "")
                    reason = doc.get("reason", "")
                    clean = filename.replace("_", " ").replace("-", " ").title()
                    st.markdown(f"- **{clean}** — {reason}")

    st.markdown("---")
    if st.button("🔙 Back to Assistant"):
        st.session_state.show_analytics = False
        st.rerun()


def show_documents_panel():
    from tools.s3_utils import get_tenant_prefix
    tenant_prefix = get_tenant_prefix()

    cfg = get_config()
    st.subheader("📁 Loaded documents")
    st.caption("Documents currently in the knowledge base")

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
            region_name=get_secret("AWS_REGION")
        )
        bucket = get_secret("S3_DOCS_BUCKET")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{tenant_prefix}/")

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


def show_gap_analysis_panel():
    st.subheader("🔎 Knowledge Base Gap Analysis")
    st.caption("AI-powered audit of document coverage, staleness, and missing content")

    if st.button("Run Gap Analysis", type="primary"):
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=get_secret("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=get_secret("AWS_SECRET_ACCESS_KEY"),
                region_name=get_secret("AWS_REGION"),
            )
            bucket = get_secret("S3_DOCS_BUCKET")

            with st.spinner("Analyzing knowledge base gaps... this may take 15-30 seconds"):
                from tools.gap_analysis import run_gap_analysis
                result = run_gap_analysis(s3, bucket)

            st.session_state["gap_analysis_result"] = result
        except Exception as e:
            st.error(f"Gap analysis failed: {e}")
            print(f"[GAP ANALYSIS] Error: {e}")
            return

    result = st.session_state.get("gap_analysis_result")
    if not result:
        st.info("Click the button above to run a full gap analysis.")
        return

    # --- Health Score ---
    score = result.get("health_score", 0)
    explanation = result.get("health_explanation", "")
    col1, col2 = st.columns([1, 3])
    col1.metric("Health Score", f"{score}/100")
    col2.markdown(f"*{explanation}*")

    # --- Coverage Gaps ---
    gaps = result.get("coverage_gaps", [])
    if gaps:
        with st.expander(f"Coverage Gaps ({len(gaps)} topics)", expanded=True):
            for gap in gaps:
                topic = gap.get("topic", "Unknown")
                suggested = gap.get("suggested_document_title", "")
                questions = gap.get("example_questions", [])
                st.markdown(f"**{topic}**")
                for q in questions:
                    st.markdown(f"- {q}")
                if suggested:
                    st.success(f"Suggested document: **{suggested}**")
                st.markdown("")

    # --- Underperforming Documents ---
    underperforming = result.get("underperforming_docs", [])
    if underperforming:
        with st.expander(f"Underperforming Documents ({len(underperforming)})"):
            for doc in underperforming:
                filename = doc.get("filename", "")
                reason = doc.get("reason", "")
                clean = filename.replace("_", " ").replace("-", " ").title()
                st.markdown(f"- **{clean}** — {reason}")

    # --- Stale Documents ---
    stale = result.get("stale_docs", [])
    if stale:
        with st.expander(f"Stale Documents ({len(stale)})"):
            for doc in stale:
                filename = doc.get("filename", "")
                days = doc.get("days_since_upload", 0)
                rec = doc.get("recommendation", "")
                clean = filename.replace("_", " ").replace("-", " ").title()
                st.markdown(f"- **{clean}** — {days} days old. {rec}")

    # --- Missing Standard Documents ---
    missing = result.get("missing_common_docs", [])
    if missing:
        with st.expander(f"Missing Standard Documents ({len(missing)})"):
            for doc in missing:
                title = doc.get("title", "")
                why = doc.get("why_needed", "")
                st.markdown(f"- **{title}** — {why}")


def show_conflict_detection_panel():
    st.subheader("⚔️ Document Conflict Detection")
    st.caption("Scans policy topics for contradictions across documents")

    if st.button("Run Conflict Detection", type="primary"):
        try:
            from tools.embeddings import load_faiss_vectorstore
            vectorstore, bm25_index = load_faiss_vectorstore("index", get_secret("OPENAI_API_KEY"))

            with st.spinner("Scanning for conflicts across 14 policy topics... this may take 30-60 seconds"):
                from tools.gap_analysis import run_conflict_detection
                conflicts = run_conflict_detection(vectorstore, bm25_index, get_secret("OPENAI_API_KEY"))

            st.session_state["conflict_detection_result"] = conflicts
        except Exception as e:
            st.error(f"Conflict detection failed: {e}")
            print(f"[CONFLICT DETECTION] Error: {e}")
            return

    conflicts = st.session_state.get("conflict_detection_result")
    if conflicts is None:
        st.info("Click the button above to scan for contradictions across documents.")
        return

    if not conflicts:
        st.success("No conflicts detected across policy topics.")
        return

    # --- Summary metrics ---
    total = len(conflicts)
    high = sum(1 for c in conflicts if c.get("severity") == "high")
    medium = sum(1 for c in conflicts if c.get("severity") == "medium")
    low = sum(1 for c in conflicts if c.get("severity") == "low")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Conflicts", total)
    col2.metric("High", high)
    col3.metric("Medium", medium)
    col4.metric("Low", low)

    # --- Display grouped by severity ---
    severity_order = ["high", "medium", "low"]
    severity_labels = {"high": "High Severity", "medium": "Medium Severity", "low": "Low Severity"}
    severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    for sev in severity_order:
        sev_conflicts = [c for c in conflicts if c.get("severity") == sev]
        if not sev_conflicts:
            continue

        icon = severity_icons[sev]
        label = severity_labels[sev]
        with st.expander(f"{icon} {label} ({len(sev_conflicts)})", expanded=(sev == "high")):
            for c in sev_conflicts:
                topic = c.get("topic", "Unknown")
                desc = c.get("description", "")
                src1 = c.get("source_1", "Unknown")
                src2 = c.get("source_2", "Unknown")
                exc1 = c.get("excerpt_1", "")
                exc2 = c.get("excerpt_2", "")

                st.markdown(f"**{topic}** — {desc}")
                col_a, col_b = st.columns(2)
                with col_a:
                    clean1 = src1.replace("_", " ").replace("-", " ").title()
                    st.caption(f"Source: {clean1}")
                    st.markdown(f"> {exc1}")
                with col_b:
                    clean2 = src2.replace("_", " ").replace("-", " ").title()
                    st.caption(f"Source: {clean2}")
                    st.markdown(f"> {exc2}")
                st.markdown("---")


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


def _show_flat_unanswered(unanswered_df):
    """Flat table fallback for unanswered questions."""
    display = (
        unanswered_df[["timestamp", "question"]]
        .sort_values("timestamp", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    display["timestamp"] = display["timestamp"].dt.strftime("%b %d, %Y %I:%M %p")
    display.columns = ["Asked on", "Question"]
    st.dataframe(display, use_container_width=True, hide_index=True)


def _cluster_questions(questions):
    """Use GPT to cluster unanswered questions into topic groups."""
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    prompt = (
        "Group these questions into 3-7 topic categories. "
        "Return JSON only, no markdown: "
        '{\"topics\": [{\"name\": string, \"questions\": string[], '
        '\"suggested_action\": string}]}. '
        "suggested_action should recommend what document to upload or update.\n\n"
        "Questions:\n" + "\n".join(f"- {q}" for q in questions)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def show_unanswered_questions(df):
    st.subheader("⚠️ Questions that need better documents")
    st.caption("These questions triggered fallback responses — consider uploading documents that cover these topics")

    if "fallback" not in df.columns:
        return

    # Filter: fallback == 1 OR gap_reason in scope
    has_gap = "gap_reason" in df.columns
    mask = df["fallback"] == 1
    if has_gap:
        mask = mask | df["gap_reason"].isin(["no_docs_retrieved", "grounding_failed"])

    unanswered = df[mask].dropna(subset=["question"]).copy()

    if unanswered.empty:
        st.success("No fallback questions — the bot is answering everything confidently.")
        return

    # Assign gap_reason for rows that only matched on fallback==1
    if has_gap:
        unanswered["gap_reason"] = unanswered["gap_reason"].fillna("unknown")
    else:
        unanswered["gap_reason"] = "unknown"

    # For fewer than 3 questions, show flat list
    if len(unanswered) < 3:
        _show_flat_unanswered(unanswered)
        return

    # --- Summary metrics ---
    total = len(unanswered)
    reason_counts = unanswered["gap_reason"].value_counts()
    most_common_reason = reason_counts.index[0]

    # --- Try clustering ---
    questions_list = unanswered["question"].tolist()
    # Build a lookup: question -> gap_reason for breakdown
    q_reasons = dict(zip(unanswered["question"], unanswered["gap_reason"]))

    try:
        with st.spinner("Clustering unanswered questions by topic..."):
            result = _cluster_questions(questions_list)
        topics = result.get("topics", [])
        if not topics:
            raise ValueError("Empty topics list")
    except Exception as e:
        print(f"[ANALYTICS] Clustering failed: {e}")
        st.warning("Could not cluster questions — showing flat list.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total unanswered", total)
        col2.metric("Most common gap reason", most_common_reason)
        col3.metric("Unique gap reasons", len(reason_counts))
        _show_flat_unanswered(unanswered)
        return

    # --- Summary row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total unanswered", total)
    col2.metric("Topic clusters", len(topics))
    col3.metric("Most common gap reason", most_common_reason)

    # --- Topic expanders ---
    for topic in topics:
        name = topic.get("name", "Unknown Topic")
        t_questions = topic.get("questions", [])
        action = topic.get("suggested_action", "")
        count = len(t_questions)

        # Gap reason breakdown for this cluster
        reasons = {}
        for q in t_questions:
            r = q_reasons.get(q, "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        with st.expander(f"{name} ({count})", expanded=False):
            for q in t_questions:
                st.markdown(f"- {q}")
            if action:
                st.info(f"**Suggested action:** {action}")
            reason_parts = [f"{r}: {c}" for r, c in sorted(reasons.items())]
            st.caption(f"Gap reasons — {', '.join(reason_parts)}")


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