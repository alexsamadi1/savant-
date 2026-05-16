from pydantic import BaseModel, Field
from typing import Optional, List, Dict


# --- Ingestion ---

class IngestResponse(BaseModel):
    tenant: str
    files_uploaded: List[str]
    schema_detected: Dict
    ready_for_analysis: bool


# --- Discovery & Analysis ---

class DiscoveryRequest(BaseModel):
    tenant: str
    company_name: str
    industry: str
    problem_statement: str
    key_questions: List[str]
    data_description: str


class AnalysisRequest(BaseModel):
    tenant: str
    discovery: DiscoveryRequest
    model: str = "gpt-4o"


class AnalysisStatus(BaseModel):
    tenant: str
    status: str = Field(..., pattern=r"^(pending|running|complete|error)$")
    progress: Optional[str] = None
    error: Optional[str] = None


# --- Dashboard ---

class MetricCard(BaseModel):
    id: str
    label: str
    value: str
    change: Optional[str] = None
    change_direction: Optional[str] = Field(
        default=None, pattern=r"^(up|down|neutral)$"
    )
    insight: Optional[str] = None


class ChartSpec(BaseModel):
    id: str
    title: str
    type: str = Field(..., pattern=r"^(bar|line|pie|scatter|area)$")
    data: List[Dict]
    x_key: str
    y_key: str
    color: Optional[str] = None
    insight: str


class Recommendation(BaseModel):
    priority: int = Field(..., ge=1, le=3)
    title: str
    detail: str
    evidence: str


class DashboardConfig(BaseModel):
    tenant: str
    company_name: str
    problem_statement: str
    generated_at: str
    executive_summary: str
    metrics: List[MetricCard]
    charts: List[ChartSpec]
    recommendations: List[Recommendation]
    data_sources_used: List[str]


# --- Chat ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    tenant: str
    messages: List[ChatMessage]
    model: str = "gpt-4o"


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    tenant: str
    has_data: bool
    has_analysis: bool
    last_updated: Optional[str] = None
