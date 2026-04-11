from pydantic import BaseModel
from typing import Optional, List

class UserProfile(BaseModel):
    role: str = "employee"
    tenure: str = "unknown"

class QueryRequest(BaseModel):
    question: str
    profile: UserProfile = UserProfile()
    model: str = "gpt-4o-mini"
    tenant: str = "demo"

class Citation(BaseModel):
    source: str
    section: Optional[str] = None
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    grounded: bool
    intent: str
    rerank_confidence: float
    latency_s: float

class UploadResponse(BaseModel):
    filename: str
    doc_count: int
    chunk_count: int

class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    tenant: str
    doc_count: int

class RebuildResponse(BaseModel):
    doc_count: int
    chunk_count: int
