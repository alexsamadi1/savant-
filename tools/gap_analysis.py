import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
from tools.s3_utils import get_secret
from tools.vectorstore_builder import get_relevant_chunks

CONFLICT_TOPICS = [
    "PTO and leave policy",
    "expense reimbursement limits",
    "travel per diem rates",
    "password and authentication requirements",
    "remote work and telework policy",
    "security clearance process",
    "timesheet submission deadlines",
    "purchase and approval thresholds",
    "dress code policy",
    "performance review schedule",
    "benefits enrollment periods and deadlines",
    "reimbursement processing timelines",
    "overtime and comp time policy",
    "onboarding and new hire procedures",
]


def run_gap_analysis(s3_client, docs_bucket, query_log_path="query_logs.csv"):
    """Analyze knowledge base gaps using document inventory and query logs."""
    from tools.s3_utils import get_tenant_prefix
    tenant_prefix = get_tenant_prefix()

    # --- 1. Document inventory from S3 ---
    response = s3_client.list_objects_v2(Bucket=docs_bucket, Prefix=f"{tenant_prefix}/")
    documents = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.endswith((".pdf", ".docx")):
                continue
            documents.append({
                "filename": key,
                "size_kb": round(obj["Size"] / 1024, 1),
                "uploaded": obj["LastModified"].strftime("%Y-%m-%d"),
                "days_since_upload": (datetime.now(obj["LastModified"].tzinfo) - obj["LastModified"]).days,
            })

    # --- 2. Query log analytics ---
    try:
        all_cols = pd.read_csv(query_log_path, nrows=0).columns.tolist()
        df = pd.read_csv(query_log_path)
        df["fallback"] = pd.to_numeric(df.get("fallback", pd.Series(dtype=float)), errors="coerce").fillna(0)
    except FileNotFoundError:
        df = pd.DataFrame()

    total_queries = len(df)

    # Fallback questions
    has_gap = "gap_reason" in df.columns if not df.empty else False
    fallback_questions = []
    gap_reason_dist = {}

    if not df.empty:
        mask = df["fallback"] == 1
        if has_gap:
            mask = mask | df["gap_reason"].isin(["no_docs_retrieved", "grounding_failed"])
        fb = df[mask].dropna(subset=["question"])
        fallback_questions = fb["question"].tolist()

        if has_gap:
            gap_reason_dist = fb["gap_reason"].fillna("unknown").value_counts().to_dict()
        else:
            gap_reason_dist = {"unknown": len(fb)}

    # Source citation counts
    most_cited = []
    least_cited = []
    if not df.empty and "source_docs" in df.columns:
        exploded = df["source_docs"].dropna().str.split(", ").explode().str.strip()
        exploded = exploded[exploded != ""]
        if not exploded.empty:
            counts = exploded.value_counts()
            most_cited = [{"doc": k, "citations": int(v)} for k, v in counts.head(10).items()]
            least_cited = [{"doc": k, "citations": int(v)} for k, v in counts.tail(10).items()]

    # --- 3. Build GPT prompt ---
    analytics = {
        "total_queries": total_queries,
        "fallback_questions": fallback_questions[:100],  # cap to stay within token limits
        "gap_reason_distribution": gap_reason_dist,
        "most_cited_documents": most_cited,
        "least_cited_documents": least_cited,
    }

    system_prompt = (
        "You are a knowledge base auditor for a government contracting organization. "
        "Given the document inventory and query analytics, produce a gap analysis report. "
        "Return JSON only, no markdown fences. Schema:\n"
        "{\n"
        '  "coverage_gaps": [{"topic": string, "example_questions": string[], '
        '"suggested_document_title": string, "regulatory_reference": string, '
        '"severity": "critical"|"moderate"|"low"}],\n'
        '  "underperforming_docs": [{"filename": string, "reason": string}],\n'
        '  "stale_docs": [{"filename": string, "days_since_upload": int, "recommendation": string}],\n'
        '  "missing_common_docs": [{"title": string, "why_needed": string}],\n'
        '  "health_score": int,\n'
        '  "health_explanation": string\n'
        "}\n\n"
        "For GovCon organizations, map each coverage gap to its "
        "specific regulatory reference where applicable "
        "(e.g. FAR 31.205-46 for travel costs, "
        "DFARS 252.204-7012 for CUI handling, "
        "DCAA ICE Model for timekeeping requirements). "
        "If no specific regulation applies, use empty string. "
        "Set severity to critical for gaps that would cause "
        "audit findings, moderate for best-practice gaps, "
        "low for nice-to-have documentation."
    )

    user_prompt = (
        f"Document inventory ({len(documents)} documents):\n"
        f"{json.dumps(documents, indent=2)}\n\n"
        f"Query analytics:\n"
        f"{json.dumps(analytics, indent=2)}"
    )

    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return json.loads(resp.choices[0].message.content)


def run_conflict_detection(vectorstore, bm25_index, api_key):
    """Check for contradictions across documents by querying high-conflict policy topics."""
    client = OpenAI(api_key=api_key)
    all_conflicts = []

    for topic in CONFLICT_TOPICS:
        docs = get_relevant_chunks(topic, vectorstore, k=5, bm25_index=bm25_index)
        if len(docs) < 2:
            continue

        # Only check if chunks come from at least 2 different sources
        sources = {doc.metadata.get("source", "") for doc in docs}
        if len(sources) < 2:
            continue

        chunks_payload = []
        for doc in docs:
            chunks_payload.append({
                "text": doc.page_content[:500],
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
            })

        system_prompt = (
            "You are a document consistency auditor. Given these excerpts about "
            f'"{topic}" from different documents, identify any contradictions, '
            "inconsistencies, or conflicting information. "
            "Return JSON only, no markdown fences. Schema:\n"
            '{"has_conflict": boolean, "conflicts": [{"description": string, '
            '"source_1": string, "source_2": string, "excerpt_1": string, '
            '"excerpt_2": string, "severity": "high"|"medium"|"low"}]}.\n'
            "If all excerpts are consistent, return {\"has_conflict\": false, \"conflicts\": []}."
        )

        user_prompt = f"Topic: {topic}\n\nExcerpts:\n{json.dumps(chunks_payload, indent=2)}"

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            result = json.loads(resp.choices[0].message.content)
            if result.get("has_conflict"):
                for conflict in result["conflicts"]:
                    conflict["topic"] = topic
                    all_conflicts.append(conflict)
        except Exception as e:
            print(f"[CONFLICT] Error checking topic '{topic}': {e}")
            continue

    return all_conflicts
