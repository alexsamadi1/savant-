import time
from datetime import datetime
from openai import OpenAI
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from config_loader import get_config
from sentence_transformers import CrossEncoder
import numpy as np

_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- Query Rewriting ---
def rewrite_query(user_input: str, client: OpenAI) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer for a GovCon knowledge base. "
                        "Rewrite the user's question into a precise search query that will retrieve the most relevant document chunks. "
                        "Expand acronyms where helpful (e.g. PTO → paid time off PTO, CUI → controlled unclassified information CUI). "
                        "Remove conversational filler. Return only the rewritten query, nothing else."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0,
            max_tokens=60
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else user_input
    except Exception:
        return user_input

# --- Rerank using Cross-Encoder ---
def rerank_chunks(query: str, chunks: List[Document]) -> List[Document]:
    if not chunks:
        return []
    pairs = [[query, chunk.page_content[:512]] for chunk in chunks]
    scores = _cross_encoder.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1]
    return [chunks[i] for i in ranked_indices]
# --- Fallback Summarization ---
def summarize_fallback(query, chunks: List[Document], client: OpenAI) -> str:
    fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in chunks[:3]])

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful knowledge assistant trained on {get_config()['brand']['company_name']}'s internal documentation. "
                "Summarize a cautious answer using the text provided. If unclear, advise the user to contact their administrator. "
                "Never fabricate organization-specific policies."
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
        return "I'm not confident I can answer that directly. Please check the source documentation or contact your administrator for guidance."

# --- Answer Revision ---
def revise_answer_with_gpt(question, draft_answer, client: OpenAI, model: str = "gpt-4o-mini") -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are editing a draft answer for clarity and tone.\n\n"
                "CRITICAL RULES:\n"
                "1. You MUST preserve all specific facts, numbers, dates, and policies from the draft\n"
                "2. Do NOT add any information that is not in the draft\n"
                "3. Do NOT replace the draft content with different information\n"
                "4. Only improve the clarity, tone, and readability\n"
                "5. Never start with a greeting like 'Hi there' or 'Hello'\n"
                "6. Never refer to the company by name — use 'your company' or 'the organization'\n"
                "7. If the draft states a specific fact or number, the final answer must preserve it exactly\n"
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
            model="gpt-4o-mini",
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
    ranked = rerank_chunks(query, docs[:3])

    if ranked:
        context = "\n\n".join([chunk.page_content[:500] for chunk in ranked])
        system_prompt = (
            f"You are {get_config()['brand']['company_name']}'s knowledge assistant. The user is a {user_profile['role']} "
            f"with {user_profile['tenure']} at the company.\n\n"
            "Your job is to clearly answer the user's question using the excerpts from internal documentation provided. "
            "If you're unsure, advise the user to contact their administrator."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {query}\n\nRelevant excerpts:\n{context}"}
        ]
    else:
        fallback_context = "\n\n".join([chunk.page_content[:500] for chunk in docs[:3]])
        messages = [
            {"role": "system", "content": (
                f"You are a helpful knowledge assistant trained on {get_config()['brand']['company_name']} internal documentation. The question wasn't answered clearly by any one excerpt, "
                "but here are some partial chunks. Summarize a helpful answer based on what you can."
            )},
            {"role": "user", "content": f"User question: {query}\n\nContext snippets:\n{fallback_context}"}
        ]

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    draft_answer = response.choices[0].message.content.strip()
    final_answer = revise_answer_with_gpt(query, draft_answer, client)

    source_doc = docs[0].metadata.get("source", "Unknown") if docs else "None"
    return final_answer, source_doc

def generate_answer(messages, client, docs=None, model: str = "gpt-4o-mini") -> Tuple[str, str, Optional[int]]:
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
            model=model,
            messages=messages
        )
        answer = response.choices[0].message.content.strip()
        return answer, source, page
    except Exception as e:
        return f"Failed to generate answer: {e}", source, page
    
def generate_answer_streaming(messages, client, docs=None, model="gpt-4o-mini"):
    """
    Call OpenAI with stream=True and return (chunk_generator, source, page).
    The generator yields text strings as they arrive.
    """
    source = "Unknown Document"
    page = None

    if docs and len(docs) > 0:
        source = docs[0].metadata.get("source", "Unknown Document")
        page = docs[0].metadata.get("page", None)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        def chunk_generator():
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        return chunk_generator(), source, page
    except Exception as e:
        def error_gen():
            yield f"Failed to generate answer: {e}"
        return error_gen(), source, page

def build_messages(user_input, context_chunk, profile, fallback=False):
    role = profile.get("role", "employee")
    tenure = profile.get("tenure", "unknown tenure")

    company = get_config()['brand']['company_name']
    today = datetime.now().strftime('%B %d, %Y')

    if fallback:
        system_prompt = (
            f"You are a helpful knowledge assistant trained on {company} internal documentation. "
            f"The user is a {role} with {tenure} at the company. "
            f"Today's date is {today}.\n\n"
            "The question wasn't answered clearly by any one excerpt, but partial context is provided. "
            "Summarize a helpful answer based on what you can. "
            "If unsure, advise the user to contact their administrator."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<context>\n{context_chunk}\n</context>\n\nQuestion: {user_input}"
            }
        ]
    else:
        system_prompt = (
            f"You are {company}'s knowledge assistant. The user is a {role} "
            f"with {tenure} at the company. "
            f"Today's date is {today}.\n\n"
            "Your job is to synthesize a clear, accurate answer using the provided context from internal documentation. "
            "If excerpts cover different aspects of the question, combine them into one cohesive answer. "
            "Be helpful and professional. If you're unsure, advise the user to contact their administrator."
        )

        if isinstance(context_chunk, list) and len(context_chunk) > 2:
            # Lost-in-middle reordering: best first, worst in middle, second-best last
            reordered = [context_chunk[0]]  # best
            reordered.extend(context_chunk[2:])  # middle ranks
            reordered.append(context_chunk[1])  # second-best at end
            context_chunk = reordered

        if isinstance(context_chunk, list):
            parts = []
            for i, chunk in enumerate(context_chunk):
                source = chunk.get("source", "Unknown Document")
                page = chunk.get("page")
                header = f"[Excerpt {i+1} from {source}, page {page}]" if page else f"[Excerpt {i+1} from {source}]"
                parts.append(f"{header}\n{chunk['text']}")
            context_text = "\n\n---\n\n".join(parts)
        else:
            source = context_chunk.get("source", "Unknown Document")
            page = context_chunk.get("page")
            source_citation = f"{source}, page {page}" if page else source
            context_text = f"[Excerpt from {source_citation}]\n{context_chunk['text']}"

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"<context>\n{context_text}\n</context>\n\nQuestion: {user_input}"
            }
        ]
    
def check_grounding(answer: str, chunks: list, client: OpenAI) -> bool:
    """
    Check if the answer is grounded in the provided chunks.
    Returns True if grounded, False if ungrounded claims detected.
    """
    context = "\n\n".join([
        f"Chunk {i+1}:\n{c['text'][:800]}"
        for i, c in enumerate(chunks)
    ])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grounding auditor. You will be given an answer and the source context chunks it was based on. "
                "Your job is to determine whether the answer contains ANY claims, facts, numbers, or policies that are NOT present in the provided context. "
                "Respond with ONLY 'yes' if ALL claims in the answer are supported by the context, or 'no' if any claim is not grounded in the context. "
                "Nothing else — just 'yes' or 'no'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Answer to audit:\n{answer}\n\n"
                f"Source context:\n{context}"
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        result = response.choices[0].message.content.strip().lower()
        return result == "yes"
    except Exception:
        return True  # If check fails, don't flag — fail open


GROUNDING_WARNING = (
    "\n\n---\n*Note: I couldn't verify all parts of this answer against your documents — "
    "please confirm with your administrator.*"
)


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
                {"role": "system", "content": "You are a helpful knowledge assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        suggestions_text = response.choices[0].message.content.strip()
        suggestions = [q.strip("•- ") for q in suggestions_text.split("\n") if q.strip()]
        return suggestions[:3]  # limit to 3 max
    except Exception as e:
        return []
