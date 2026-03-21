import time
from openai import OpenAI
from typing import List, Tuple, Optional
from langchain.schema.document import Document
from config_loader import get_config

# --- Rerank using GPT ---
def rerank_with_gpt(query, chunks, client: OpenAI) -> Optional[str]:
    if not chunks:
        return None

    context_snippets = "\n\n".join([
        f"Chunk {i+1} (Source: {chunk.metadata.get('source', 'unknown')}):\n{chunk.page_content[:600]}"
        for i, chunk in enumerate(chunks)
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are selecting the single best chunk to answer a user's question.\n\n"
                "Instructions:\n"
                "- Read the question carefully\n"
                "- Read each chunk carefully\n"
                "- Select the chunk whose content most directly answers the question\n"
                "- Chunk 1 should be selected if it contains relevant information — do not skip it\n"
                "- Respond with ONLY the number of the best chunk, e.g. '1' or '2'\n"
                "- If truly no chunk answers the question, respond with '0'\n"
                "- Nothing else — just the number"
            )
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nChunks:\n{context_snippets}"
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        content = response.choices[0].message.content.strip()

        import re
        match = re.search(r'\d+', content)
        if match:
            chunk_index = int(match.group()) - 1
            if chunk_index < 0:
                return None
            if 0 <= chunk_index < len(chunks):
                return chunks[chunk_index].page_content
        return None

    except Exception:
        return None
# --- Fallback Summarization ---
def summarize_fallback(query, chunks: List[Document], client: OpenAI) -> str:
    fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in chunks[:3]])

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful assistant trained on {get_config()['brand']['company_name']}'s employee handbook and onboarding documents. "
                "Summarize a cautious answer using the text provided. If unclear, advise contacting HR. "
                "Never fabricate company-specific policies."
            )
        },
        {
            "role": "user",
            "content": f"User question: {query}\n\nPartial content:\n{fallback_context}"
        }
    ]

    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        return response.choices[0].message.content.strip()
    except Exception:
        return "I'm not confident I can answer that directly. Please check the handbook or contact HR for guidance."

# --- Answer Revision ---
def revise_answer_with_gpt(question, draft_answer, client: OpenAI) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are editing a draft HR answer for clarity and tone.\n\n"
                "CRITICAL RULES:\n"
                "1. You MUST preserve all specific facts, numbers, dates, and policies from the draft\n"
                "2. Do NOT add any information that is not in the draft\n"
                "3. Do NOT replace the draft content with different information\n"
                "4. Only improve the clarity, tone, and readability\n"
                "5. Never start with a greeting like 'Hi there' or 'Hello'\n"
                "6. Never refer to the company by name — use 'your company' or 'the organization'\n"
                "7. If the draft says 15 PTO days, the final answer must say 15 PTO days\n"
                "8. Keep the same length — do not expand or summarize drastically"
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Draft answer to improve (keep all facts exactly):\n{draft_answer}"
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        revised = response.choices[0].message.content.strip()
        
        # Safety check — if revised answer is completely different length it went wrong
        # Fall back to draft in that case
        if len(revised) < len(draft_answer) * 0.4:
            return draft_answer
            
        return revised
    except Exception:
        return draft_answer

# --- Unified Response Generator ---
def generate_response(
    query: str,
    docs: List[Document],
    client: OpenAI,
    user_profile: dict
) -> Tuple[str, str]:
    """
    Returns: (final_answer, source_title)
    """
    chunks = docs[:3]
    reranked = rerank_with_gpt(query, chunks, client)

    if reranked:
        system_prompt = (
            f"You are {get_config()[‘brand’][‘company_name’]}’s professional HR assistant. The user is a {user_profile[‘role’]} "
            f"with {user_profile[‘tenure’]} at the company.\n\n"
            "Your job is to clearly answer the user's HR question using the excerpt provided. "
            "If you're unsure, advise the user to contact HR."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {query}\n\nRelevant excerpt:\n{reranked}"}
        ]
    else:
        fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in chunks])
        messages = [
            {"role": "system", "content": (
                f"You are a helpful HR assistant trained on {get_config()[‘brand’][‘company_name’]} documents. The question wasn’t answered clearly by any one excerpt, "
                "but here are some partial chunks. Summarize a helpful answer based on what you can."
            )},
            {"role": "user", "content": f"User question: {query}\n\nContext snippets:\n{fallback_context}"}
        ]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    draft_answer = response.choices[0].message.content.strip()
    final_answer = revise_answer_with_gpt(query, draft_answer, client)

    source_doc = docs[0].metadata.get("source", "Unknown") if docs else "None"
    return final_answer, source_doc

def generate_answer(messages, client, docs=None) -> Tuple[str, str, Optional[int]]:
    """
    Call OpenAI and return (answer, source, page).
    source and page are extracted from the first doc's metadata if provided.
    """
    source = "Unknown Document"
    page = None

    if docs and len(docs) > 0:
        source = docs[0].metadata.get("source", "Unknown Document")
        page = docs[0].metadata.get("page", None)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        return answer, source, page
    except Exception as e:
        return f"Failed to generate answer: {e}", source, page
    
def build_messages(user_input, context_chunk, profile, fallback=False):
    role = profile.get("role", "employee")
    tenure = profile.get("tenure", "unknown tenure")

    if fallback:
        return [
            {
                "role": "system",
                "content": (
                    f"You are a helpful HR assistant trained on {get_config()[‘brand’][‘company_name’]} documents. "
                    "The question wasn’t answered clearly by any one excerpt, but here are some partial chunks. "
                    "Summarize a helpful answer based on what you can.\n"
                    "If unsure, advise the user to contact HR."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {user_input}\n\n"
                    f"Context snippets:\n{context_chunk}"
                )
            }
        ]
    else:
        source = context_chunk.get("source", "Unknown Document")
        page = context_chunk.get("page")
        source_citation = f"{source}, page {page}" if page else source

        return [
            {
                "role": "system",
                "content": (
                    f"You are {get_config()[‘brand’][‘company_name’]}’s professional HR assistant. The user is a {role} "
                    f"with {tenure} at the company.\n\n"
                    "Your job is to clearly answer the user's HR question using the excerpt provided. "
                    "Be helpful and professional. If you're unsure, advise the user to contact HR."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User question: {user_input}\n\n"
                    f"Relevant excerpt (from {source_citation}):\n\n{context_chunk['text']}"
                )
            }
        ]
    
def suggest_follow_ups(user_question, answer, client: OpenAI) -> list:
    prompt = (
        f"Based on the following user question and assistant answer, suggest 2 to 3 helpful follow-up questions "
        f"that the user might ask next. Keep them concise and relevant.\n\n"
        f"User Question: {user_question}\n\n"
        f"Assistant Answer: {answer}\n\n"
        f"Follow-up Suggestions:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        suggestions_text = response.choices[0].message.content.strip()
        suggestions = [q.strip("•- ") for q in suggestions_text.split("\n") if q.strip()]
        return suggestions[:3]  # limit to 3 max
    except Exception as e:
        return []
