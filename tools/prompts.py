from config_loader import get_config

def build_system_prompt(company: str, role: str, tenure: str, today: str, system_prompt_layer: str = "") -> str:
    base = (
        f"You are {company}'s knowledge assistant. The user is a {role} "
        f"with {tenure} at the company. "
        f"Today's date is {today}.\n\n"
        "Your job is to synthesize a clear, accurate answer using the provided "
        "context from internal documentation. "
        "If excerpts cover different aspects of the question, combine them into "
        "one cohesive answer. Be helpful and professional. "
        "If you're unsure, advise the user to contact their administrator."
    )
    formatting_rules = (
        "\n\nFormatting rules — always follow these:\n"
        "- Lead with the direct answer on the first line, "
        "in bold if it is a short phrase or key finding\n"
        "- Use bullet points for any list of 3+ items, "
        "programs, findings, or steps\n"
        "- Bold key terms, names, dates, dollar amounts, "
        "and percentages\n"
        "- Keep paragraphs to 2-3 sentences maximum\n"
        "- If comparing multiple items, use a structured "
        "list not prose\n"
        "- End with the source document name and date "
        "if known\n"
        "- Never start with 'Based on the documents', "
        "'According to', or similar preamble — "
        "lead with the answer"
    )
    full_prompt = f"{base}{formatting_rules}"
    return (
        f"{full_prompt}\n\n{system_prompt_layer}".strip()
        if system_prompt_layer
        else full_prompt
    )

def build_fallback_system_prompt(company: str, role: str, tenure: str, today: str) -> str:
    return (
        f"You are a helpful knowledge assistant trained on {company} internal "
        f"documentation. The user is a {role} with {tenure} at the company. "
        f"Today's date is {today}.\n\n"
        "The question wasn't answered clearly by any one excerpt, but partial "
        "context is provided. Summarize a helpful answer based on what you can. "
        "If unsure, advise the user to contact their administrator."
    )

def build_prompt(query: str, documents: list, role: str = None, tenure: str = None) -> str:
    company_name = get_config()["brand"]["company_name"]

    context_blocks = []
    for doc in documents:
        title = doc.metadata.get("section_title", "Unknown Section")
        source = doc.metadata.get("source", "")
        block = f"[{title} | {source}]\n{doc.page_content}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    user_context = ""
    if role and tenure:
        user_context = f"The user is a {role} who has been with {company_name} for {tenure}.\n"

    return f"""You are a {company_name} knowledge assistant. {user_context}Use only the following context from the official {company_name} internal documentation to answer.

If the answer is not clearly in the provided context, respond with: "I couldn't find that in the documentation. Please check with your administrator."

Context:
{context}

Question:
{query}
"""
