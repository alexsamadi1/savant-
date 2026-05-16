"""Chat route: streaming follow-up Q&A against analysis context."""

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.models import ChatRequest
from logic.chat_logic import build_messages, generate_answer_streaming
from tools.s3_utils import get_s3_client, get_secret

router = APIRouter()


def _load_dashboard_context(tenant: str) -> str:
    """Load dashboard.json and build a context string for the chat system prompt."""
    s3 = get_s3_client()
    bucket = get_secret("S3_DOCS_BUCKET")
    try:
        resp = s3.get_object(Bucket=bucket, Key=f"{tenant}/dashboard.json")
        dashboard = json.loads(resp["Body"].read())
    except Exception:
        raise HTTPException(status_code=404, detail=f"No analysis found for tenant '{tenant}'")

    # Build context from executive summary and recommendations
    parts = [f"Executive Summary:\n{dashboard.get('executive_summary', '')}"]

    recs = dashboard.get("recommendations", [])
    if recs:
        parts.append("Recommendations:")
        for r in recs:
            parts.append(f"- [{r.get('priority', '?')}] {r.get('title', '')}: {r.get('detail', '')}")

    metrics = dashboard.get("metrics", [])
    if metrics:
        parts.append("Key Metrics:")
        for m in metrics:
            parts.append(f"- {m.get('label', '')}: {m.get('value', '')} ({m.get('insight', '')})")

    return "\n\n".join(parts)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream GPT responses as SSE events."""
    context = _load_dashboard_context(request.tenant)

    # Convert ChatMessage models to dicts for the history (exclude latest message)
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
    user_input = request.messages[-1].content

    messages = build_messages(user_input, context, conversation_history=history)

    async def event_generator():
        try:
            for token in generate_answer_streaming(messages, model=request.model):
                yield {"data": json.dumps({"token": token, "done": False})}
            yield {"data": json.dumps({"token": None, "done": True})}
        except Exception as e:
            yield {"data": json.dumps({"token": f"Error: {e}", "done": True})}

    return EventSourceResponse(event_generator())
