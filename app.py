import streamlit as st
from openai import OpenAI
from tools.embeddings import load_faiss_vectorstore
from tools.s3_utils import get_secret
from tools.s3_utils import upload_file_to_s3
from tools.vectorstore_builder import rebuild_vectorstore_from_s3, get_relevant_chunks
from tools.log_utils import ensure_log_file_exists, log_chat_interaction
from tools.analytics_dashboard import show_analytics_dashboard
from tools.filename_generator import generate_smart_filename, extract_text_from_docx
from logic.chat_logic import generate_answer, build_messages
from config_loader import get_config
from io import BytesIO
from pathlib import Path
import uuid
import time
import re
import os
import nltk

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
  display: inline-block;
  font-size: 0.75rem;
  background-color: #0d1f1e;
  color: #00C9A7;
  border: 1px solid #00C9A7;
  border-radius: 20px;
  padding: 2px 10px;
  margin-top: 6px;
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

# --- User Onboarding ---
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

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
    except Exception as e:
        st.warning(f"⚠️ Couldn't load vectorstore from S3. Rebuilding... ({e})")
        rebuild_vectorstore_from_s3()
        return load_faiss_vectorstore("index", get_secret("OPENAI_API_KEY"))

vectorstore, bm25_index = get_vectorstore()

# --- OpenAI Client ---
client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        st.markdown(f"<div class='chat-bubble {bubble}'>{entry['content']}</div>", unsafe_allow_html=True)

# --- Handle User Input ---
user_input = st.chat_input(assistant["chat_placeholder"])

if "example_question" in st.session_state and not user_input:
    user_input = st.session_state.pop("example_question")

if not user_input or not isinstance(user_input, str) or not user_input.strip():
    st.stop()

st.chat_message("user").markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)
st.session_state.chat_history.append({"role": "user", "content": user_input})

# --- Generate Response ---
with st.spinner("Searching documents..."):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='chat-bubble bot-bubble'>🤖 Typing<span class='dots'></span></div>",
            unsafe_allow_html=True
        )

        docs = get_relevant_chunks(user_input, vectorstore, k=20, bm25_index=bm25_index)

        if not docs:
            answer = assistant["fallback_message"]
            placeholder.markdown(f"<div class='chat-bubble bot-bubble'>{answer}</div>", unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            log_chat_interaction(user_input, answer, profile, [], fallback=True, response_type="summary")
            st.stop()

        top_chunks = [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page")
            }
            for doc in docs[:5]
        ]
        messages = build_messages(user_input, top_chunks, profile, fallback=False)

        # --- Generate answer (now returns tuple) ---
        answer, source, page = generate_answer(messages, client, docs=docs, model=st.session_state.get("selected_model", "gpt-4o-mini"))

        # --- Stream answer ---
        streamed_response = ""
        for i, char in enumerate(answer):
            streamed_response += char
            cursor = "▌" if i % 2 == 0 else ""
            placeholder.markdown(
                f"<div class='chat-bubble bot-bubble'>{streamed_response}{cursor}</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.008)

        placeholder.markdown(
            f"<div class='chat-bubble bot-bubble'>{answer}</div>",
            unsafe_allow_html=True
        )

        # --- Citation chip ---
        if source and source != "Unknown Document":
            clean_source = source.replace("_", " ").strip().title()
            section_title = docs[0].metadata.get("section_title", "") if docs else ""
            if section_title and section_title != "Introduction":
                citation_label = f"📄 {clean_source} — {section_title}"
            elif page:
                citation_label = f"📄 {clean_source} — Page {page}"
            else:
                citation_label = f"📄 {clean_source}"
            st.markdown(f"<div class='citation-chip'>{citation_label}</div>", unsafe_allow_html=True)
        # --- Scroll to bottom ---
        st.markdown("<div id='bottom'></div>", unsafe_allow_html=True)
        st.markdown("""
            <script>
                const bottom = document.getElementById("bottom");
                if (bottom) bottom.scrollIntoView({behavior: "smooth"});
            </script>
        """, unsafe_allow_html=True) 
        # --- Log ---
        source_titles = [doc.metadata.get("source", "unknown") for doc in docs]
        log_chat_interaction(user_input, answer, profile, source_titles, fallback=False, response_type="direct")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
