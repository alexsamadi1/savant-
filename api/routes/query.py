import time
import os
import json
from fastapi import APIRouter, Depends
from openai import OpenAI
from api.models import QueryRequest, QueryResponse, Citation
from api.dependencies import get_openai_client, get_vectorstore
from logic.chat_logic import rewrite_query, rerank_chunks, build_messages, generate_answer, generate_answer_streaming, check_grounding
from tools.vectorstore_builder import get_relevant_chunks
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, client: OpenAI = Depends(get_openai_client)):
    os.environ["TENANT_PREFIX"] = req.tenant
    vectorstore, bm25_index = get_vectorstore(req.tenant)

    start = time.time()
    rewritten, intent = rewrite_query(req.question, client)
    k = 30 if intent == "synthesis" else 8
    docs = get_relevant_chunks(rewritten, vectorstore, k=k, bm25_index=bm25_index)

    if not docs:
        return QueryResponse(
            answer="I couldn't find relevant information in your documents.",
            citations=[], grounded=False, intent=intent,
            rerank_confidence=0.0, latency_s=round(time.time() - start, 2)
        )

    ranked, confidence = rerank_chunks(rewritten, docs[:10])
    top_chunks = [
        {"text": d.page_content, "source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in ranked[:5]
    ]
    messages = build_messages(rewritten, top_chunks, {"role": req.profile.role, "tenure": req.profile.tenure})
    answer, source, page = generate_answer(messages, client, docs=ranked, model=req.model)
    is_grounded = check_grounding(answer, top_chunks, client)

    seen = set()
    citations = []
    for doc in ranked[:5]:
        src = doc.metadata.get("source", "Unknown")
        if src in seen:
            continue
        seen.add(src)
        if "/" in src:
            src = src.split("/", 1)[-1]
        citations.append(Citation(
            source=src.replace("_", " ").strip().title(),
            section=doc.metadata.get("section_title"),
            page=doc.metadata.get("page"),
        ))
        if len(citations) >= 3:
            break

    return QueryResponse(
        answer=answer, citations=citations,
        grounded=is_grounded, intent=intent,
        rerank_confidence=round(confidence, 4),
        latency_s=round(time.time() - start, 2),
    )


@router.post("/query/stream")
async def query_stream(req: QueryRequest, client: OpenAI = Depends(get_openai_client)):
    os.environ["TENANT_PREFIX"] = req.tenant
    vectorstore, bm25_index = get_vectorstore(req.tenant)

    rewritten, intent = rewrite_query(req.question, client)
    k = 30 if intent == "synthesis" else 8
    docs = get_relevant_chunks(rewritten, vectorstore, k=k, bm25_index=bm25_index)

    if not docs:
        async def no_docs():
            yield {"data": json.dumps({"token": None, "done": True,
                "answer": "I couldn't find relevant information.",
                "citations": [], "grounded": False})}
        return EventSourceResponse(no_docs())

    ranked, confidence = rerank_chunks(rewritten, docs[:10])
    top_chunks = [
        {"text": d.page_content, "source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for d in ranked[:5]
    ]
    messages = build_messages(rewritten, top_chunks, {"role": req.profile.role, "tenure": req.profile.tenure})
    stream_gen, source, page = generate_answer_streaming(messages, client, docs=ranked, model=req.model)

    seen = set()
    citations = []
    for doc in ranked[:5]:
        src = doc.metadata.get("source", "Unknown")
        if src in seen:
            continue
        seen.add(src)
        if "/" in src:
            src = src.split("/", 1)[-1]
        citations.append({
            "source": src.replace("_", " ").strip().title(),
            "section": doc.metadata.get("section_title"),
            "page": doc.metadata.get("page"),
        })
        if len(citations) >= 3:
            break

    async def token_stream():
        full = ""
        for token in stream_gen:
            full += token
            yield {"data": json.dumps({"token": token, "done": False})}
        is_grounded = check_grounding(full, top_chunks, client)
        yield {"data": json.dumps({
            "token": None, "done": True,
            "answer": full,
            "citations": citations,
            "grounded": is_grounded,
            "intent": intent,
            "rerank_confidence": round(confidence, 4),
        })}

    return EventSourceResponse(token_stream())
