import re
from openai import OpenAI
import streamlit as st
from docx import Document

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_smart_filename(document_text: str, original_name: str = "") -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You're an assistant that generates clean, descriptive filenames from document content. "
                "Use lowercase and underscores. Make it concise and relevant to HR topics like onboarding, PTO, telework, benefits, etc. "
                "No special characters or punctuation. Respond with only the filename base (no extension)."
            )
        },
        {
            "role": "user",
            "content": f"Here's the document content:\n\n{document_text[:1500]}\n\nWhat should the filename be?"
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        raw = response.choices[0].message.content.strip()
        clean = re.sub(r"[^\w\d_]", "", raw.lower().replace(" ", "_"))
        return f"{clean or 'uploaded_file'}.docx"
    except Exception as e:
        print(f"[Filename GPT Error] {e}")
        fallback = original_name.replace(" ", "_").lower() or "uploaded_file"
        return f"{fallback}.docx"
    
def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])