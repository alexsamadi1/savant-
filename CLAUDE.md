Here it is — the final version with all updates baked in. Copy this exactly:
markdown# CLAUDE.md — Savant 2.0

This file is read by Claude Code at the start of every session. Do not delete it.

---

## Context

We are working on branch `savant-2` of the existing Savant repository.
Savant 1.0 (the RAG chatbot) is preserved on `main`. Do not touch `main`.
Before writing any code, confirm the current branch with `git branch`.
Commit after each phase: `git add . && git commit -m "phase X: description"`

---

## What This Project Is

Savant 2.0 is an AI-assisted consulting platform. It is NOT a general-purpose RAG chatbot (that was Savant 1.0).

The product flow:
1. Client fills out a discovery form (company info, problem statement, 3-5 key questions)
2. Client uploads structured data (CSV, Excel) and optional documents (PDF, DOCX)
3. An AI agent analyzes everything using SQL, Python, and RAG tools
4. A dashboard is generated: executive summary, KPI cards, charts, recommendations
5. A chat interface lets the client ask follow-up questions against the analysis

**V1 scope is strict.** Do not add features not listed here. When in doubt, ask before building.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.11 |
| Frontend | Next.js 14 (App Router) + TypeScript |
| Data storage | SQLite (per-tenant, stored in S3) |
| File storage | AWS S3 (boto3) |
| Vector search | FAISS + LangChain (documents only) |
| AI | OpenAI GPT-4o via official SDK |
| Deployment | Railway (single shared deployment) |
| Monitoring | Sentry |
| Charts | Recharts (already in web/package.json) |

---

## Project Structure
savant-2/
├── api/
│   ├── main.py              # FastAPI app entrypoint — DO NOT RESTRUCTURE
│   ├── dependencies.py      # Shared deps (tenant resolution) — DO NOT MODIFY
│   ├── models.py            # All Pydantic request/response models
│   └── routes/
│       ├── health.py        # GET /health — DO NOT MODIFY
│       ├── ingest.py        # POST /ingest/data, POST /ingest/documents
│       ├── analyze.py       # POST /analyze, GET /analyze/status/{tenant}
│       ├── dashboard.py     # GET /dashboard/{tenant}
│       └── chat.py          # POST /chat/stream
├── logic/
│   ├── analysis_agent.py        # Core agent with 4 tools
│   ├── data_loader.py           # CSV/Excel → pandas → SQLite → S3
│   ├── schema_detective.py      # GPT-powered schema suggestions
│   ├── doc_ingestor.py          # PDF/DOCX → FAISS index → S3
│   ├── doc_structure_extractor.py  # GPT extracts metadata → SQLite row
│   └── chat_logic.py            # Stripped: build_messages + generate_answer_streaming only
├── tools/
│   ├── s3_utils.py          # S3 helpers + get_secret() — DO NOT MODIFY
│   └── log_utils.py         # Logging helpers — DO NOT MODIFY
├── clients/
│   └── *.toml               # Legacy Savant 1.0 tenant configs — PRESERVE FILES, IGNORE CONTENTS
├── config.toml              # App-level config
├── config_loader.py         # Config loader — DO NOT MODIFY
├── requirements.txt
└── web/                     # Next.js frontend
└── src/app/
├── globals.css          # Design system — USE THESE VARIABLES ONLY
├── layout.tsx           # Root layout — DO NOT RESTRUCTURE
├── Sidebar.tsx          # Nav — update nav items only
├── page.tsx             # Root — redirects to /onboarding
├── onboarding/          # Discovery form + file upload (4 steps)
├── dashboard/[tenant]/  # Rendered analysis output
└── chat/[tenant]/       # Follow-up chat

---

## Secrets & Config

All secrets are retrieved via `get_secret()` from `tools/s3_utils.py`.
Never hardcode secrets. Never use `os.environ` directly.

Required secrets:
- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_DOCS_BUCKET`
- `S3_INDEX_BUCKET`

S3 key structure per tenant:
{tenant}/data.db                # SQLite database (CSV data + document metadata)
{tenant}/schema.json            # Detected schema from uploaded files
{tenant}/faiss/                 # FAISS index directory (if docs uploaded)
{tenant}/dashboard.json         # Final dashboard output
{tenant}/analysis_status.json   # running | complete | error

---

## Tenant System

- Auto-generated from company name on onboarding form
- Format: lowercase, hyphens, no special chars ("Acme Corp" → "acme-corp")
- Append 4-char random suffix if collision risk
- No authentication in v1 — clients access via direct URL
- `clients/` directory has legacy Savant 1.0 configs — preserve files but the new config structure in `config.toml` takes precedence

---

## Document Ingestion — Dual Pipeline (Critical)

Every uploaded document runs TWO simultaneous pipelines:

| Pipeline | File | Output | Powers |
|---|---|---|---|
| FAISS indexing | `doc_ingestor.py` | Vector index in S3 | `rag_search` tool — retrieve passages |
| Structure extraction | `doc_structure_extractor.py` | Row in `documents` SQLite table | `sql_query` tool — analyze patterns across corpus |

The `documents` SQLite table has one row per uploaded document:
`filename`, `doc_date`, `word_count`, `has_executive_summary`, `has_risk_section`,
`has_milestones`, `has_deliverables`, `has_budget_section`, `section_count`,
`tone_score`, `specificity_score`, `completeness_score`, `key_topics`

**Why this matters:** RAG retrieves passages from single documents. SQL analyzes patterns across all documents. Both are required. "Which reports are missing risk sections?" is a SQL query, not a RAG search.

- Use `gpt-4o-mini` for structure extraction (runs per-document at ingest — cost matters)
- Use `response_format={"type": "json_object"}` to guarantee parseable output
- `key_topics` is stored as a JSON string in SQLite (arrays don't fit SQL cells)

---

## Data Layer Rules

- CSV/Excel → pandas → SQLite table, stored as `{tenant}/data.db` in S3
- Multiple file uploads APPEND new tables — do NOT overwrite the DB
- Table names derived from filenames: `sales_q1.csv` → table `sales_q1`
- Agent queries via `sql_query` tool (SELECT only — no DDL in agent)
- Text truncated to 4,000 chars for GPT structure extraction

---

## Analysis Agent Rules

Located at `logic/analysis_agent.py`. Has exactly 4 tools:

| Tool | Purpose |
|---|---|
| `sql_query` | SELECT queries against tenant's SQLite (CSV data + documents table) |
| `rag_search` | Semantic search over uploaded documents via FAISS |
| `python_exec` | pandas/numpy for complex calculations |
| `generate_chart_spec` | Produce Recharts-compatible chart config |

- Max 20 iterations
- Runs as a FastAPI background task
- Chart specs are collected as they are generated during the loop
- Final message must be valid JSON matching `DashboardConfig` schema
- Status polled via `GET /analyze/status/{tenant}`
- Model: `gpt-4o` for analysis, `gpt-4o-mini` for schema detection and doc extraction

---

## Frontend Rules

**Styling:**
- Use ONLY CSS variables from `globals.css` — no Tailwind, no inline hex colors
- Key variables: `--teal`, `--surface`, `--bg`, `--text-primary`, `--text-secondary`, `--border`

**Charts:**
- Recharts only (already installed)
- Map `ChartSpec.type` → `BarChart`, `LineChart`, `AreaChart`, `PieChart`
- Wrap all charts: `<ResponsiveContainer width="100%" height={300}>`

**Streaming:**
- Reuse `web/src/lib/useStream.ts` for SSE
- Chat endpoint: `POST /chat/stream`

**Navigation:**
/onboarding          → New Engagement (discovery form + upload)
/dashboard/{tenant}  → Analysis dashboard
/chat/{tenant}       → Follow-up chat

---

## What NOT To Build in V1

Do not build these even if they seem useful or the client asks:

- Live dashboard chart updates from chat
- User authentication / login flows
- Multi-user collaboration or permissions
- Automated API connectors (HubSpot, QuickBooks, Salesforce) — CSV/Excel only
- White-labeling or custom domains
- Billing or subscription management
- Email notifications
- Export to PDF or PowerPoint

---

## File Conventions

- Python: snake_case, type hints on all function signatures, docstrings on all public functions
- TypeScript: PascalCase components, camelCase variables, explicit return types
- No `any` in TypeScript without a comment explaining why
- All API routes return typed Pydantic models
- Async FastAPI routes: `async def` — background tasks: regular `def`

---

## Key Design Decisions (Do Not Revisit Without Asking)

1. **Branch `savant-2`** — all work here; `main` is frozen as Savant 1.0
2. **Single Railway deployment** — tenant isolation via S3 key prefix, not separate services
3. **SQLite over PostgreSQL** — simplicity for v1; each tenant's data is self-contained
4. **Agent runs once** — analysis cached in S3 as JSON; not real-time
5. **No auth in v1** — shareable URL is sufficient for early clients
6. **FAISS over Pinecone** — already in codebase, no new external dependency
7. **Dual-path document ingestion** — FAISS for retrieval, SQLite for analytics

---

## Session Start Checklist

Before writing any code:
1. Run `git branch` — confirm you are on `savant-2`
2. Read this file fully
3. Know which phase you are working on (Implementation Guide in Notion)
4. Know which files to preserve vs. modify
5. Do not add features outside v1 scope
6. Commit after each phase completes successfully