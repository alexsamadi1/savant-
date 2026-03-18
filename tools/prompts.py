def build_prompt(query: str, documents: list, role: str = None, tenure: str = None) -> str:
    context_blocks = []
    for doc in documents:
        title = doc.metadata.get("section_title", "Unknown Section")
        source = doc.metadata.get("source", "")
        block = f"[{title} | {source}]\n{doc.page_content}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    user_context = ""
    if role and tenure:
        user_context = f"The user is a {role} who has been with Innovim for {tenure}.\n"

    return f"""You are an Innovim HR assistant. {user_context}Use only the following context from the official Innovim Employee Handbook and Orientation Guide to answer.

If the answer is not clearly in the provided context, respond with: “I couldn’t find that in the handbook. Please check with HR.”

Context:
{context}

Question:
{query}
"""