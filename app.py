import sentry_sdk
try:
    from tools.s3_utils import get_secret as _get_secret_init
    sentry_sdk.init(
        dsn=_get_secret_init("SENTRY_DSN"),
        traces_sample_rate=0.1,
        environment="production",
    )
    print("[SENTRY] Initialized successfully")
except Exception as e:
    print(f"[SENTRY] Not initialized: {e}")

import streamlit as st
from openai import OpenAI
from tools.embeddings import load_faiss_vectorstore
from tools.s3_utils import get_secret
from tools.s3_utils import upload_file_to_s3
from tools.vectorstore_builder import rebuild_vectorstore_from_s3, get_relevant_chunks
from tools.log_utils import ensure_log_file_exists, log_chat_interaction
from tools.analytics_dashboard import show_analytics_dashboard
from tools.filename_generator import generate_smart_filename, extract_text_from_docx
from logic.chat_logic import generate_answer, generate_answer_streaming, build_messages, check_grounding, GROUNDING_WARNING, rerank_chunks, rewrite_query, suggest_follow_ups
from config_loader import get_config
from io import BytesIO
from pathlib import Path
import uuid
import time
import re
import html
import os
import json
import nltk
from streamlit_js_eval import streamlit_js_eval

# --- Load Config ---
cfg = get_config()
brand = cfg["brand"]
contact = cfg["contact"]
onboarding = cfg["onboarding"]
assistant = cfg["assistant"]

# --- Page Setup ---
st.set_page_config(
    page_title=brand["app_name"],
    page_icon=brand["page_icon"],
    layout="wide"
)
ensure_log_file_exists()

# --- NLTK Setup ---
def ensure_nltk_resources(resources):
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'taggers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

ensure_nltk_resources(['punkt', 'averaged_perceptron_tagger'])

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# --- Global CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: #e0e0e0;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background-color: #0a0a0f;
}

.main .block-container {
  background-color: #0a0a0f;
  padding-top: 2rem;
  padding-bottom: 2rem;
  max-width: 860px;
}

[data-testid="stSidebar"] {
  background-color: #111118;
  border-right: 1px solid rgba(0, 201, 167, 0.15);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
  color: #e0e0e0;
}
[data-testid="stSidebar"] .stCaption {
  color: #666666 !important;
}

.chat-bubble {
  margin: 0.5rem 0;
  padding: 1rem 1.25rem;
  border-radius: 18px;
  display: inline-block;
  max-width: 90%;
  line-height: 1.6;
  transition: all 0.2s ease;
}
.user-bubble {
  background-color: #00C9A7;
  color: #ffffff;
  align-self: flex-end;
  box-shadow: 0 0 18px rgba(0, 201, 167, 0.35);
}
.bot-bubble {
  background-color: #0d1f1e;
  color: #e0e0e0;
  align-self: flex-start;
  border-left: 3px solid #00C9A7;
}

.citation-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  font-weight: 500;
  background-color: rgba(0, 201, 167, 0.1);
  color: #00C9A7;
  border: 1px solid rgba(0, 201, 167, 0.4);
  border-radius: 8px;
  padding: 8px 14px;
  margin-top: 8px;
  margin-bottom: 4px;
  letter-spacing: 0.01em;
}

[data-testid="stButton"] button {
  background-color: #0d1f1e;
  color: #e0e0e0;
  border: 1px solid #00C9A7;
  border-radius: 8px;
  transition: all 0.2s ease;
}
[data-testid="stButton"] button:hover {
  background-color: #00C9A7;
  color: #ffffff;
  box-shadow: 0 0 14px rgba(0, 201, 167, 0.5);
  border-color: #00C9A7;
}

/* Chat input container + text */
div[data-testid="stChatInput"] {
  background-color: #0d1f1e !important;
  border: 1px solid rgba(0, 201, 167, 0.3) !important;
  border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea {
  background-color: #0d1f1e !important;
  color: #ffffff !important;
  border-radius: 12px;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: #00C9A7;
  box-shadow: 0 0 10px rgba(0, 201, 167, 0.2);
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #555 !important;
}

/* Onboarding heading */
.onboarding-heading {
  text-align: center;
  background: linear-gradient(90deg, #00C9A7, #00f5d4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
}

/* Radio label text */
div[data-testid="stRadio"] label {
  color: #ffffff !important;
}
div[data-testid="stRadio"] label p {
  color: #ffffff !important;
}

/* Radio options — dark pill style */
div[role="radiogroup"] label {
  background-color: #0d1f1e !important;
  border: 1px solid rgba(0, 201, 167, 0.25) !important;
  border-radius: 8px !important;
  padding: 0.5rem 1rem !important;
  margin: 0.2rem 0 !important;
  color: #ffffff !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
}
div[role="radiogroup"] label:hover {
  border-color: #00C9A7 !important;
}
div[role="radiogroup"] label:has(input:checked) {
  border-color: #00C9A7 !important;
  background-color: rgba(0, 201, 167, 0.15) !important;
}

/* Primary button — Continue */
button[data-testid="baseButton-primary"] {
  background-color: #00C9A7 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  box-shadow: 0 0 20px rgba(0, 201, 167, 0.35) !important;
  transition: all 0.2s ease !important;
}
button[data-testid="baseButton-primary"]:hover {
  background-color: #00a88c !important;
  box-shadow: 0 0 28px rgba(0, 201, 167, 0.55) !important;
}

/* Sample questions expander */
[data-testid="stExpander"] {
  border: 1px solid rgba(0, 201, 167, 0.2) !important;
  border-radius: 10px !important;
  background-color: #111118 !important;
}
[data-testid="stExpander"] summary {
  color: #e0e0e0 !important;
}
[data-testid="stExpander"] summary:hover {
  color: #00C9A7 !important;
}

.dots { display: inline-block; width: 1em; text-align: left; }
.dots::after {
  content: '...';
  animation: dotsAnim 1.5s steps(3, end) infinite;
}
@keyframes dotsAnim {
  0%   { content: ''; }
  33%  { content: '.'; }
  66%  { content: '..'; }
  100% { content: '...'; }
}
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- localStorage Persistence ---
MAX_CHAT_MESSAGES = 50

if "ls_loaded" not in st.session_state:
    st.session_state.ls_loaded = False

if not st.session_state.ls_loaded:
    # JS returns None until the browser executes it and triggers a rerun.
    # Use "__empty__" sentinel so we can tell "JS hasn't run" (None) from
    # "localStorage has no value" ("__empty__").
    saved_profile = streamlit_js_eval(
        js_expressions="JSON.parse(localStorage.getItem('savant_profile') || '\"__empty__\"')",
        key="ls_profile",
    )
    saved_chat = streamlit_js_eval(
        js_expressions="JSON.parse(localStorage.getItem('savant_chat_history') || '\"__empty__\"')",
        key="ls_chat",
    )

    if saved_profile is None:
        # JS hasn't executed yet — show spinner, stop, wait for auto-rerun
        st.markdown(
            "<div style='display:flex;justify-content:center;align-items:center;height:60vh;'>"
            "<p style='color:#888;font-size:1.1rem;'>Loading your profile…</p></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # JS has returned — process values and mark done
    st.session_state.ls_loaded = True
    if isinstance(saved_profile, dict) and "role" in saved_profile and "tenure" in saved_profile:
        st.session_state.user_profile = saved_profile
    if isinstance(saved_chat, list):
        st.session_state.chat_history = saved_chat[-MAX_CHAT_MESSAGES:]


def save_profile_to_ls(profile_dict):
    """Save user profile to localStorage."""
    profile_json = json.dumps(profile_dict)
    streamlit_js_eval(js_expressions=f"localStorage.setItem('savant_profile', JSON.stringify({profile_json})), null", key=f"save_profile_{time.time()}")


def save_chat_to_ls(chat_list):
    """Save chat history to localStorage (truncated to last MAX_CHAT_MESSAGES)."""
    truncated = chat_list[-MAX_CHAT_MESSAGES:]
    chat_json = json.dumps(truncated)
    streamlit_js_eval(js_expressions=f"localStorage.setItem('savant_chat_history', JSON.stringify({chat_json})), null", key=f"save_chat_{time.time()}")


# --- User Onboarding ---
profile = st.session_state.user_profile

if "role" not in profile or "tenure" not in profile:
    st.markdown(
        "<h2 class='onboarding-heading'>👋 Welcome! Let's get to know you first</h2>",
        unsafe_allow_html=True
    )

    st.session_state.role_selection = st.radio(
        onboarding["role_question"],
        onboarding["role_options"],
        key="role_radio"
    )
    st.session_state.tenure_selection = st.radio(
        onboarding["tenure_question"],
        onboarding["tenure_options"],
        key="tenure_radio"
    )

    if st.button("✅ Continue", type="primary", use_container_width=True):
        profile["role"] = st.session_state.role_selection
        profile["tenure"] = st.session_state.tenure_selection
        save_profile_to_ls(profile)
        st.success("You're all set! You can now start asking questions below 👇")
        st.rerun()
    else:
        st.stop()

if st.session_state.get("show_analytics", False):
    show_analytics_dashboard()
    st.stop()

# --- Load Vectorstore ---
@st.cache_resource(show_spinner="🔍 Loading knowledge base...")
def get_vectorstore():
    try:
        return load_faiss_vectorstore("index", get_secret("OPENAI_API_KEY"))
    except FileNotFoundError:
        st.error("Knowledge base not found. Please contact your administrator.")
        st.stop()
    except Exception as e:
        st.error("Could not load the knowledge base. Please try refreshing the page.")
        print(f"[ERROR] Vectorstore load failed: {e}")
        st.stop()

vectorstore, bm25_index = get_vectorstore()

# --- OpenAI Client ---
client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

# --- Sidebar ---
if "role" in profile and "tenure" in profile:
    with st.sidebar:
        st.image(brand["logo_path"], use_container_width=True)
        st.markdown(f"### {brand['sidebar_title']}")

        role = profile.get("role", "Unknown Role")
        tenure = profile.get("tenure", "Unknown Tenure")
        st.markdown(f"**👤 Role:** {role}")
        st.markdown(f"**📆 Tenure:** {tenure}")
        st.caption(f"_{brand['sidebar_caption']}_")

        models_cfg = get_config()["models"]
        if "selected_model" not in st.session_state:
            st.session_state["selected_model"] = models_cfg["default"]
        model_options = models_cfg["options"]
        model_labels = [opt["label"] for opt in model_options]
        model_values = [opt["value"] for opt in model_options]
        current_index = model_values.index(st.session_state["selected_model"]) if st.session_state["selected_model"] in model_values else 0
        selected_label = st.selectbox("🤖 Model", model_labels, index=current_index)
        st.session_state["selected_model"] = model_values[model_labels.index(selected_label)]
        st.caption("Swap models anytime — won't reset your chat")

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            streamlit_js_eval(js_expressions="localStorage.removeItem('savant_chat_history'), null", key="clear_chat_ls")
            st.rerun()
        if st.button("🔄 Reset Profile", use_container_width=True):
            st.session_state.user_profile = {}
            st.session_state.chat_history = []
            st.session_state.ls_loaded = False
            streamlit_js_eval(js_expressions="localStorage.removeItem('savant_profile'), localStorage.removeItem('savant_chat_history'), null", key="reset_profile_ls")
            st.rerun()

        with st.expander("ℹ️ How to Use & Support", expanded=False):
            st.markdown("- Ask clear, specific questions")
            st.markdown("- Answers come from your organization's internal documents")
            st.markdown("- Every answer shows its source")
            st.markdown("---")
            st.markdown(f"[📣 Submit Feedback]({contact['feedback_url']})")

        with st.expander("🔒 Admin Tools"):
            admin_code = st.text_input("Admin code", type="password")
            if admin_code == get_secret("ADMIN_CODE"):
                st.session_state.is_admin = True
                st.success("Admin access granted")

            if st.session_state.get("is_admin", False):
                uploaded_file = st.file_uploader("Upload doc (.pdf/.docx)", type=["pdf", "docx"])
                if uploaded_file:
                    if uploaded_file.name != st.session_state.get("last_uploaded_file"):
                        try:
                            raw_text = extract_text_from_docx(uploaded_file) if uploaded_file.name.endswith(".docx") else ""
                            smart_filename = generate_smart_filename(raw_text, uploaded_file.name)
                            upload_file_to_s3(BytesIO(uploaded_file.getbuffer()), smart_filename, get_secret("S3_DOCS_BUCKET"))
                            st.success(f"Uploaded as `{smart_filename}`")

                            with st.spinner("Rebuilding knowledge base... this takes 1-2 minutes"):
                                from tools.vectorstore_builder import rebuild_vectorstore_enriched
                                doc_count, chunk_count = rebuild_vectorstore_enriched()
                                st.success(f"✅ Knowledge base updated — {doc_count} documents, {chunk_count} chunks indexed")

                            st.session_state.last_uploaded_file = uploaded_file.name
                            st.cache_resource.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {e}")
                    else:
                        st.info("File already uploaded.")

                if st.button("🔄 Rebuild Knowledge Base"):
                    with st.spinner("Rebuilding knowledge base from S3... this takes 1-2 minutes"):
                        from tools.vectorstore_builder import rebuild_vectorstore_enriched
                        doc_count, chunk_count = rebuild_vectorstore_enriched()
                    st.success(f"✅ Knowledge base rebuilt — {doc_count} documents, {chunk_count} chunks indexed")
                    st.cache_resource.clear()

                if st.button("📊 Open Dashboard"):
                    st.session_state.show_analytics = True

        st.markdown(
            f"<div style='font-size: 0.75rem; color: gray; margin-top: 1rem;'>{brand['footer_text']}</div>",
            unsafe_allow_html=True
        )

# --- Main Header ---
st.markdown(f"<h1 style='text-align: center;'>{brand['app_name']} Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888; font-size: 1rem;'>Your go-to assistant for institutional knowledge, policies, and procedures.</p>", unsafe_allow_html=True)

# --- Sample Questions ---
examples = assistant["sample_questions"]
with st.expander("💡 Try a sample question", expanded=False):
    cols = st.columns(len(examples))
    for i, q in enumerate(examples):
        with cols[i]:
            if st.button(f"👉 {q}", key=f"sample_{i}"):
                st.session_state["example_question"] = q
# --- Empty State ---
if not st.session_state.chat_history and "example_question" not in st.session_state:
    with st.chat_message("assistant"):
        topics_html = "".join([f"<li>{t}</li>" for t in assistant["topics"]])
        st.markdown(f"""
        <div class='chat-bubble bot-bubble'>
            {assistant['welcome_message']}
            <ul>{topics_html}</ul>
            Just type your question below or click one of the samples to get started.
        </div>
        """, unsafe_allow_html=True)

# --- Chat History Display ---
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        bubble = "user-bubble" if entry["role"] == "user" else "bot-bubble"
        content = html.escape(entry['content']) if entry["role"] == "user" else entry['content']
        st.markdown(f"<div class='chat-bubble {bubble}'>{content}</div>", unsafe_allow_html=True)

# --- Handle User Input ---
user_input = st.chat_input(assistant["chat_placeholder"])

if "example_question" in st.session_state and not user_input:
    user_input = st.session_state.pop("example_question")

if "followup_question" in st.session_state and not user_input:
    user_input = st.session_state.pop("followup_question")

if not user_input or not isinstance(user_input, str) or not user_input.strip():
    st.stop()

MAX_QUERY_LENGTH = 1000

if user_input and len(user_input) > MAX_QUERY_LENGTH:
    st.warning(f"Your question is too long ({len(user_input)} characters). Please keep it under {MAX_QUERY_LENGTH} characters.")
    st.stop()

if user_input:
    user_input = " ".join(user_input.split())

st.chat_message("user").markdown(f"<div class='chat-bubble user-bubble'>{html.escape(user_input)}</div>", unsafe_allow_html=True)
st.session_state.chat_history.append({"role": "user", "content": user_input})
save_chat_to_ls(st.session_state.chat_history)

# --- Generate Response ---
with st.spinner("Searching documents..."):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='chat-bubble bot-bubble'>🤖 Typing<span class='dots'></span></div>",
            unsafe_allow_html=True
        )

        try:
            start_time = time.time()
            rewritten = rewrite_query(user_input, client)
            docs = get_relevant_chunks(rewritten, vectorstore, k=30, bm25_index=bm25_index)

            if not docs:
                answer = assistant["fallback_message"]
                placeholder.markdown(f"<div class='chat-bubble bot-bubble'>⚠️ {answer}</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='font-size: 0.8rem; color: #666; margin-top: 4px; padding-left: 8px;'>"
                    "💡 Tip: If you think this should be covered, let your admin know — they can upload the relevant document.</div>",
                    unsafe_allow_html=True
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                save_chat_to_ls(st.session_state.chat_history)
                log_chat_interaction(user_input, answer, profile, [], fallback=True, response_type="fallback", gap_reason="no_docs_retrieved")
                st.stop()

            ranked, rerank_confidence = rerank_chunks(rewritten, docs[:10])
            print(f"[RERANK] confidence={rerank_confidence:.3f} — {user_input[:80]}")
            top_chunks = [
                {
                    "text": doc.page_content,
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
                for doc in ranked[:5]
            ]
            # Extract last 3 exchanges for multi-turn context
            conv_history = []
            history = st.session_state.chat_history[:-1]  # exclude the just-appended user msg
            pairs = []
            i = len(history) - 1
            while i >= 1 and len(pairs) < 3:
                if history[i]["role"] == "assistant" and history[i - 1]["role"] == "user":
                    pairs.append((history[i - 1], history[i]))
                    i -= 2
                else:
                    i -= 1
            for user_msg, asst_msg in reversed(pairs):
                conv_history.append({"role": "user", "content": user_msg["content"]})
                conv_history.append({"role": "assistant", "content": asst_msg["content"][:500]})

            messages = build_messages(rewritten, top_chunks, profile, fallback=False, conversation_history=conv_history)

            # --- Stream answer ---
            stream_gen, source, page = generate_answer_streaming(
                messages, client, docs=ranked,
                model=st.session_state.get("selected_model", "gpt-4o-mini")
            )

            streamed_response = ""
            for token in stream_gen:
                streamed_response += token
                placeholder.markdown(
                    f"<div class='chat-bubble bot-bubble'>{streamed_response}▌</div>",
                    unsafe_allow_html=True
                )

            answer = streamed_response

            # Final render without cursor
            placeholder.markdown(
                f"<div class='chat-bubble bot-bubble'>{answer}</div>",
                unsafe_allow_html=True
            )

            latency = round(time.time() - start_time, 2)
            print(f"[LATENCY] {latency}s — {user_input[:80]}")

            # --- Citation chips (up to 3 deduplicated sources) ---
            seen_sources = set()
            for doc in ranked[:5]:
                src = doc.metadata.get("source", "Unknown Document")
                section = doc.metadata.get("section_title", "")
                pg = doc.metadata.get("page")
                if src == "Unknown Document" or src in seen_sources:
                    continue
                seen_sources.add(src)
                clean_src = src.replace("_", " ").strip().title()
                if section and section != "Introduction":
                    label = f"📄 {clean_src} — {section}"
                elif pg:
                    label = f"📄 {clean_src} — Page {pg}"
                else:
                    label = f"📄 {clean_src}"
                st.markdown(f"<div class='citation-chip'>{label}</div>", unsafe_allow_html=True)
                if len(seen_sources) >= 3:
                    break
            # --- Feedback ---
            feedback_key = f"feedback_{len(st.session_state.chat_history)}"
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"{feedback_key}_up", help="Helpful answer"):
                    print(f"[FEEDBACK] 👍 — {user_input[:80]}")
            with col2:
                if st.button("👎", key=f"{feedback_key}_down", help="Needs improvement"):
                    print(f"[FEEDBACK] 👎 — {user_input[:80]}")

            # --- Scroll to bottom ---
            st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
            st.markdown("""
                <script>
                    const bottom = document.getElementById("bottom");
                    if (bottom) bottom.scrollIntoView({behavior: "smooth"});
                </script>
            """, unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            save_chat_to_ls(st.session_state.chat_history)

            # --- Grounding check (runs after answer is visible) ---
            gap_reason = "direct"
            fallback_phrases = [
                "do not include", "do not contain", "does not include", "does not contain",
                "couldn't find", "cannot find", "no specific", "not specifically",
                "not mentioned", "provided excerpts", "excerpts provided"
            ]
            if not any(phrase in answer.lower() for phrase in fallback_phrases):
                is_grounded = check_grounding(answer, top_chunks, client)
                if not is_grounded:
                    gap_reason = "grounding_failed"
                    answer += GROUNDING_WARNING
                    placeholder.markdown(
                        f"<div class='chat-bubble bot-bubble'>{answer}</div>",
                        unsafe_allow_html=True
                    )
                    st.session_state.chat_history[-1]["content"] = answer
                    save_chat_to_ls(st.session_state.chat_history)

            # --- Log ---
            source_titles = [doc.metadata.get("source", "unknown") for doc in docs]
            log_chat_interaction(user_input, answer, profile, source_titles, fallback=(gap_reason != "direct"), response_type="direct", gap_reason=gap_reason)

            # --- Follow-up suggestions ---
            try:
                followups = suggest_follow_ups(user_input, answer, client)
                if followups:
                    cols = st.columns(len(followups))
                    for i, suggestion in enumerate(followups):
                        with cols[i]:
                            if st.button(suggestion, key=f"followup_{len(st.session_state.chat_history)}_{i}"):
                                st.session_state["followup_question"] = suggestion
                                st.rerun()
            except Exception as e:
                print(f"[FOLLOWUP] Error generating suggestions: {e}")

        except Exception as e:
            sentry_sdk.capture_exception(e)
            print(f"[ERROR] Pipeline failure: {type(e).__name__}: {e}")
            error_msg = "⚠️ Something went wrong while generating your answer. Please try again."
            placeholder.markdown(
                f"<div class='chat-bubble bot-bubble'>{error_msg}</div>",
                unsafe_allow_html=True
            )
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            save_chat_to_ls(st.session_state.chat_history)
            log_chat_interaction(user_input, error_msg, profile, [], fallback=True, response_type="error", gap_reason="error")
