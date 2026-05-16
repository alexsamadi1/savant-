"""Chat logic: message building and streaming answer generation."""

from typing import Generator, List, Optional

from openai import OpenAI

from tools.s3_utils import get_secret

SYSTEM_PROMPT = (
    "You are Savant, an AI analyst. Answer questions using only the provided analysis context. "
    "Be specific and cite metrics and recommendations where relevant."
)


def build_messages(
    user_input: str,
    context: str,
    conversation_history: Optional[List[dict]] = None,
) -> List[dict]:
    """Build the OpenAI messages array for a chat turn.

    Args:
        user_input: The user's current question.
        context: The analysis context to inject (e.g. dashboard JSON summary).
        conversation_history: Optional list of prior messages in OpenAI format.

    Returns:
        A list of message dicts ready to pass to the OpenAI API.
    """
    messages: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Analysis context:\n{context}"},
    ]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_input})

    return messages


def generate_answer_streaming(
    messages: List[dict],
    model: str = "gpt-4o",
) -> Generator[str, None, None]:
    """Stream an answer from the OpenAI API token by token.

    Args:
        messages: The messages array built by build_messages.
        model: The OpenAI model to use.

    Yields:
        Each token string as it arrives from the stream.
    """
    client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            yield token
