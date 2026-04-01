"""
Savant Golden Eval Runner
=========================
Runs the 30-question golden eval set against the live RAG pipeline.
Outputs a results CSV with actual answers, latency, and citation pre-filled.
You fill in quality_score (0, 0.5, 1) and citation_correct (0, 1) manually.
RAGAS auto-scores are computed for faithfulness, relevancy, precision, recall.

Usage:
    python run_eval.py

Requirements:
    - Run from the root of your savant repo
    - .streamlit/secrets.toml must exist with OPENAI_API_KEY, AWS keys, etc.
    - FAISS index must be built locally in faiss_index/

Output:
    eval_results_YYYYMMDD_HHMMSS.csv
"""

import csv
import time
import os
import sys
import toml
from datetime import datetime
from pathlib import Path

print("Savant Eval Runner starting...")

# ---------------------------------------------------------------------------
# Bootstrap — load secrets directly from toml, no Streamlit dependency
# ---------------------------------------------------------------------------
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

secrets_path = repo_root / ".streamlit" / "secrets.toml"
if not secrets_path.exists():
    print(f"ERROR: secrets.toml not found at {secrets_path}")
    sys.exit(1)

_secrets = toml.load(secrets_path)

# Inject into environment so boto3, OpenAI, dotenv all pick them up
for k, v in _secrets.items():
    if isinstance(v, str):
        os.environ[k] = v

print("Secrets loaded")

# ---------------------------------------------------------------------------
# Monkey-patch get_secret before any other imports touch it
# Prevents the Streamlit import path from being triggered at module level
# in log_utils.py and analytics_dashboard.py
# ---------------------------------------------------------------------------
import tools.s3_utils as s3_mod

def _patched_get_secret(key: str) -> str:
    val = _secrets.get(key) or os.environ.get(key)
    if not val:
        raise ValueError(f"Secret '{key}' not found in secrets.toml")
    return val

s3_mod.get_secret = _patched_get_secret
print("get_secret patched")

# ---------------------------------------------------------------------------
# Import pipeline components
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    print("OpenAI imported")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    from tools.embeddings import load_faiss_vectorstore
    print("embeddings imported")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    from tools.vectorstore_builder import get_relevant_chunks
    print("vectorstore_builder imported")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    from logic.chat_logic import rerank_chunks, rewrite_query, build_messages, generate_answer
    print("chat_logic imported")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    from config_loader import get_config
    print("config_loader imported")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_CSV = "savant_eval_set_v2.csv"
K_RETRIEVAL = 30   # matches app.py


# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
def load_pipeline():
    print("\nLoading pipeline...")
    api_key = _patched_get_secret("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    print("Loading vectorstore...")
    # load_faiss_vectorstore returns (faiss_vectorstore, bm25_index) in current codebase
    result = load_faiss_vectorstore("index", api_key)
    if isinstance(result, tuple):
        vectorstore, bm25_index = result
    else:
        vectorstore = result
        bm25_index = None

    cfg = get_config()
    model = cfg["models"]["default"]
    print(f"Pipeline ready \u2014 model: {model}")
    if bm25_index is None:
        print("WARNING: BM25 index not found \u2014 running FAISS-only retrieval")
    else:
        print("BM25 hybrid search active")

    return client, vectorstore, bm25_index, cfg


# ---------------------------------------------------------------------------
# Run a single query through the full pipeline
# ---------------------------------------------------------------------------
def run_query(question: str, client, vectorstore, bm25_index, cfg) -> dict:
    start = time.time()

    # Step 1: Rewrite query
    rewritten = rewrite_query(question, client)

    # Step 2: Retrieve
    docs = get_relevant_chunks(rewritten, vectorstore, k=K_RETRIEVAL, bm25_index=bm25_index)

    if not docs:
        latency = round(time.time() - start, 2)
        return {
            "actual_answer": "[NO DOCS RETRIEVED \u2014 fallback triggered]",
            "latency_s": latency,
            "citation": "None",
            "source": "none",
            "chunks_retrieved": 0,
            "rerank_hit": False,
            "contexts": [],
        }

    # Step 3: Rerank top 10
    ranked = rerank_chunks(rewritten, docs[:10])
    rerank_hit = len(ranked) > 0

    # Step 4: Build messages — pass list of top 5 chunks (matches app.py)
    top_chunks = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in ranked[:5]
    ]
    profile = {"role": "Evaluator", "tenure": "N/A"}
    messages = build_messages(rewritten, top_chunks, profile, fallback=False)

    # Step 5: Generate
    model = cfg["models"]["default"]
    answer, source, page = generate_answer(messages, client, docs=ranked, model=model)

    latency = round(time.time() - start, 2)

    # Step 6: Build citation (matches app.py logic)
    section_title = ranked[0].metadata.get("section_title", "") if ranked else ""
    clean_source = (source or "unknown").replace("_", " ").strip().title()
    if section_title and section_title != "Introduction":
        citation = f"{clean_source} \u2014 {section_title}"
    elif page:
        citation = f"{clean_source} \u2014 Page {page}"
    else:
        citation = clean_source if clean_source not in ("Unknown", "Unknown Document") else "No citation"

    # Collect top-5 retrieved contexts for RAGAS scoring
    contexts = [doc.page_content for doc in docs[:5]]

    return {
        "actual_answer": answer,
        "latency_s": latency,
        "citation": citation,
        "source": source or "unknown",
        "chunks_retrieved": len(docs),
        "rerank_hit": rerank_hit,
        "contexts": contexts,
    }


# ---------------------------------------------------------------------------
# RAGAS auto-scoring
# ---------------------------------------------------------------------------
def run_ragas_eval(results: list) -> list:
    """Run RAGAS metrics on eval results. Returns per-question score dicts."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        print(f"\n  WARNING: RAGAS import failed ({e}) \u2014 skipping auto-scoring")
        return []

    try:
        # Build HuggingFace Dataset in the format RAGAS expects
        data = {
            "question": [r["question"] for r in results],
            "answer": [r["actual_answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["expected_answer"] for r in results],
        }
        dataset = Dataset.from_dict(data)

        # Use gpt-4o-mini — NOT gpt-4
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        print("\n  Running RAGAS evaluation (this may take a few minutes)...")
        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
        )

        # Convert to per-question score list
        df = ragas_result.to_pandas()
        scores = []
        for _, row in df.iterrows():
            f = row.get("faithfulness", None)
            ar = row.get("answer_relevancy", None)
            cp = row.get("context_precision", None)
            cr = row.get("context_recall", None)

            # Compute combined as average of available scores
            available = [s for s in [f, ar, cp, cr] if s is not None and s == s]  # filter NaN
            combined = sum(available) / len(available) if available else None

            scores.append({
                "ragas_faithfulness": round(f, 4) if f is not None and f == f else "",
                "ragas_answer_relevancy": round(ar, 4) if ar is not None and ar == ar else "",
                "ragas_context_precision": round(cp, 4) if cp is not None and cp == cp else "",
                "ragas_context_recall": round(cr, 4) if cr is not None and cr == cr else "",
                "ragas_combined": round(combined, 4) if combined is not None else "",
            })

        print("  RAGAS scoring complete.")
        return scores

    except Exception as e:
        print(f"\n  WARNING: RAGAS evaluation failed ({e}) \u2014 skipping auto-scoring")
        return []


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
def run_eval():
    eval_path = Path(EVAL_CSV)
    if not eval_path.exists():
        print(f"\nERROR: Eval CSV not found: {EVAL_CSV}")
        print("Make sure savant_eval_set_v2.csv is in the repo root.")
        sys.exit(1)

    with open(eval_path, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    print(f"\nLoaded {len(questions)} questions from {EVAL_CSV}")

    client, vectorstore, bm25_index, cfg = load_pipeline()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(".") / f"eval_results_{timestamp}.csv"

    results = []
    total_latency = 0.0
    slow_queries = []
    rerank_misses = []

    print(f"\n{'\u2500'*72}")
    print(f"{'Q':>3}  {'Latency':>8}  {'Rerank':>6}  {'Question':<50}")
    print(f"{'\u2500'*72}")

    for row in questions:
        q_id = row["id"]
        question = row["question"]
        expected = row["expected_answer"]
        difficulty = row["difficulty"]
        q_type = row["question_type"]
        is_adversarial = q_type == "adversarial"

        try:
            result = run_query(question, client, vectorstore, bm25_index, cfg)
        except Exception as e:
            result = {
                "actual_answer": f"[ERROR: {e}]",
                "latency_s": 0.0,
                "citation": "error",
                "source": "error",
                "chunks_retrieved": 0,
                "rerank_hit": False,
                "contexts": [],
            }
            print(f"  Q{q_id} error: {e}")

        total_latency += result["latency_s"]
        if result["latency_s"] > 5.0:
            slow_queries.append(q_id)
        if not result["rerank_hit"]:
            rerank_misses.append(q_id)

        rerank_indicator = "HIT" if result["rerank_hit"] else "MISS"
        q_short = question[:48] + ".." if len(question) > 50 else question
        print(f"{q_id:>3}  {result['latency_s']:>7.2f}s  {rerank_indicator:>6}  {q_short:<50}")

        notes = ""
        if is_adversarial:
            notes = "ADVERSARIAL: score 1.0 if clean fallback, 0.0 if fabricated answer"
        elif q_type == "ambiguous":
            notes = "AMBIGUOUS: score 1.0 if conservative policy answer, 0.5 if hedged"

        results.append({
            "id": q_id,
            "difficulty": difficulty,
            "question_type": q_type,
            "question": question,
            "expected_answer": expected,
            "actual_answer": result["actual_answer"],
            "citation": result["citation"],
            "source_retrieved": result["source"],
            "chunks_retrieved": result["chunks_retrieved"],
            "rerank_hit": result["rerank_hit"],
            "latency_s": result["latency_s"],
            "quality_score": "",
            "citation_correct": "",
            "notes": notes,
            "contexts": result["contexts"],
        })

    print(f"{'\u2500'*72}")

    # -----------------------------------------------------------------------
    # Run RAGAS auto-scoring
    # -----------------------------------------------------------------------
    ragas_scores = run_ragas_eval(results)

    # Merge RAGAS scores into results
    empty_ragas = {
        "ragas_faithfulness": "",
        "ragas_answer_relevancy": "",
        "ragas_context_precision": "",
        "ragas_context_recall": "",
        "ragas_combined": "",
    }
    for i, r in enumerate(results):
        scores = ragas_scores[i] if i < len(ragas_scores) else empty_ragas
        r.update(scores)
        # Remove contexts — not needed in CSV
        r.pop("contexts", None)

    fieldnames = [
        "id", "difficulty", "question_type", "question",
        "expected_answer", "actual_answer",
        "citation", "source_retrieved",
        "chunks_retrieved", "rerank_hit",
        "latency_s", "quality_score", "citation_correct", "notes",
        "ragas_faithfulness", "ragas_answer_relevancy",
        "ragas_context_precision", "ragas_context_recall", "ragas_combined",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    avg_latency = round(total_latency / len(questions), 2)
    rerank_hit_count = sum(1 for r in results if r["rerank_hit"])

    print(f"\n{'='*72}")
    print(f"  DONE \u2014 {len(questions)} questions")
    print(f"  Output: {output_path}")
    print(f"  Avg latency: {avg_latency}s")
    print(f"  Reranker hit: {rerank_hit_count}/{len(questions)}")
    if slow_queries:
        print(f"  Slow queries (>5s): Q{', Q'.join(slow_queries)}")
    if rerank_misses:
        print(f"  Reranker misses: Q{', Q'.join(rerank_misses)}")
    print(f"{'='*72}")

    # -----------------------------------------------------------------------
    # RAGAS aggregate summary
    # -----------------------------------------------------------------------
    if ragas_scores:
        def _mean(key):
            vals = [s[key] for s in ragas_scores if s[key] != "" and s[key] is not None]
            return round(sum(vals) / len(vals) * 100, 1) if vals else None

        mean_f = _mean("ragas_faithfulness")
        mean_ar = _mean("ragas_answer_relevancy")
        mean_cp = _mean("ragas_context_precision")
        mean_cr = _mean("ragas_context_recall")
        mean_combined = _mean("ragas_combined")

        print(f"\n  RAGAS Scores (auto)")
        print(f"  {'\u2500'*40}")
        print(f"  Faithfulness:       {mean_f}%" if mean_f is not None else "  Faithfulness:       N/A")
        print(f"  Answer Relevancy:   {mean_ar}%" if mean_ar is not None else "  Answer Relevancy:   N/A")
        print(f"  Context Precision:  {mean_cp}%" if mean_cp is not None else "  Context Precision:  N/A")
        print(f"  Context Recall:     {mean_cr}%" if mean_cr is not None else "  Context Recall:     N/A")
        print(f"  {'\u2500'*40}")
        print(f"  Combined (avg):     {mean_combined}%" if mean_combined is not None else "  Combined (avg):     N/A")
        print()
        if mean_combined is not None:
            if mean_combined >= 90.0:
                print(f"  >> {mean_combined}% >= 90% \u2014 Phase 3 ready")
            elif mean_combined >= 80.0:
                print(f"  ~~ {mean_combined}% >= 80% \u2014 Phase 2a threshold met")
            else:
                print(f"  XX {mean_combined}% < 80% \u2014 below Phase 2a threshold")
        print(f"{'='*72}")

    print(f"""
  Next steps:
  1. Open {output_path}
  2. Fill in 'quality_score':   0 = wrong  |  0.5 = partial  |  1 = correct
  3. Fill in 'citation_correct': 0 = wrong/missing  |  1 = correct
  4. Query success rate = SUM(quality_score) / {len(questions)}
  5. Citation accuracy  = SUM(citation_correct) / {len(questions)}

  Adversarial Qs (Q25-29): score 1.0 ONLY if system said it could not find
  the answer. Score 0.0 if it returned a confident fabricated policy answer.

  RAGAS columns are auto-filled \u2014 compare ragas_combined vs manual quality_score
  to calibrate the automated metric.
""")

if __name__ == "__main__":
    run_eval()

    # Auto-archive to S3
try:
    import boto3
    s3 = boto3.client("s3",
        aws_access_key_id=_patched_get_secret("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_patched_get_secret("AWS_SECRET_ACCESS_KEY"),
        region_name=_patched_get_secret("AWS_REGION")
    )
    s3_key = f"evals/{output_path.name}"
    s3.upload_file(str(output_path), _patched_get_secret("S3_DOCS_BUCKET"), s3_key)
    print(f"  Archived to S3: {s3_key}")
except Exception as e:
    print(f"  S3 archive skipped: {e}")
