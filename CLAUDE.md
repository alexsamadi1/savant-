# CLAUDE.md

T# Savant — Claude Context File

## Commands

**Run the app:**
```bash
streamlit run app.py
```

**Deploy (Heroku/cloud):**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

No test suite or linter is configured.

---

## Architecture

Savant is a RAG (Retrieval-Augmented Generation) knowledge assistant built with Streamlit + OpenAI + FAISS.

### Request Flow

1. User submits query via Streamlit chat input
2. FAISS vectorstore returns top-10 similar document chunks (`tools/vectorstore_builder.py:get_relevant_chunks`)
3. GPT reranks to the single best chunk (`logic/chat_logic.py:rerank_with_gpt`)
4. System prompt is built with user's role/tenure profile (`tools/prompts.py`)
5. GPT generates a draft answer (`logic/chat_logic.py:generate_answer`)
6. GPT revises the draft for clarity/tone (`logic/chat_logic.py:revise_answer_with_gpt`)
7. Answer streams to the UI character-by-character with a citation (source document + page)
8. Interaction is logged to CSV and uploaded to S3 (`tools/log_utils.py`)

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI: onboarding, chat loop, admin panel, CSS |
| `logic/chat_logic.py` | Core RAG functions: rerank, generate, revise |
| `tools/vectorstore_builder.py` | Build/rebuild FAISS index from S3 documents |
| `tools/embeddings.py` | Load FAISS index (S3 → local fallback); `@st.cache_resource` |
| `tools/loaders.py` | PDF/DOCX parsing with section extraction |
| `tools/s3_utils.py` | All S3 read/write operations |
| `tools/log_utils.py` | CSV interaction logging |
| `tools/analytics_dashboard.py` | Admin analytics UI (query logs, user demographics) |
| `config.toml` | Brand, onboarding questions, assistant topics, S3 bucket names, model options |
| `config_loader.py` | Loads `config.toml` at startup |

### Configuration

All app-level settings (brand name, model list, onboarding questions, S3 bucket, sample questions) live in `config.toml`. Runtime secrets are in `.streamlit/secrets.toml` (not committed):

- `OPENAI_API_KEY`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
- `S3_DOCS_BUCKET`, `S3_INDEX_BUCKET`
- `ADMIN_CODE` (unlocks admin upload + analytics panel)

Access secrets via `st.secrets.get_secret("KEY")` — never hardcode or read from environment directly.

### Vectorstore

- FAISS index stored locally at `faiss_index/` (gitignored) and backed up to S3
- On startup, `load_faiss_vectorstore()` pulls the index from S3 if no local copy exists
- Admins can trigger a full rebuild (re-embed all S3 documents) from the sidebar
- Two rebuild paths: `rebuild_vectorstore_from_s3()` (basic) and `rebuild_vectorstore_enriched()` (adds section metadata)

### Models

Users can select between `gpt-4o-mini` (fast/default) and `gpt-4o` (smart) from the sidebar. The selected model is passed through the entire RAG chain. Model list and default are configured in `config.toml [models]`.

### Admin Features

Unlocked by entering `ADMIN_CODE` in the sidebar:
- Upload PDF/DOCX → auto-uploaded to S3 and triggers vectorstore rebuild
- View analytics dashboard (query volume, user demographics, unanswered questions)
- Manual vectorstore rebuild button

---

## Project Goals & Current Focus

- **Company:** Savant — early-stage AI, targeting government contractors
- **Core product:** Knowledge retrieval + workflow automation on top of institutional documents
- **Immediate goal:** First paying customer
- **Current sprint:** [UPDATE WEEKLY — e.g., "Building RAG eval harness + fixing single-chunk reranker"]
- I am the solo technical founder — no handoffs, just ship

---

## My Conventions

- Keep functions small and modular — one job per function
- All prompts go in `tools/prompts.py`, never inline in logic files
- Never hardcode keys — always use `st.secrets`
- Prefer readable code over clever code
- When adding a new feature, touch the fewest files possible

---

## Known Weaknesses (Active TODO)

- Reranker currently returns only 1 chunk — should pass top 3 into generation
- No eval harness yet — RAG accuracy is unmeasured
- Still on OpenAI — Claude API migration is planned
- No test suite configured

---

## How to Help Me (Claude Code Behavior)

- You are an executor, not an architect — I handle design decisions in Claude chat
- Make the change I ask for, in the fewest files possible
- Don't refactor things I didn't ask you to touch
- If something is ambiguous, ask one clarifying question before writing code
- Never modify prompts in `tools/prompts.py` unless explicitly told to
- Flag breaking changes to S3/FAISS pipeline before implementing, not after
- No explanations unless I ask — just show me the diff