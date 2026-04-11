export interface UserProfile { role: string; tenure: string; }
export interface Citation { source: string; section?: string; page?: number; }
export interface QueryResponse { answer: string; citations: Citation[]; grounded: boolean; intent: string; rerank_confidence: number; latency_s: number; }
export interface StreamEvent { token: string | null; done: boolean; answer?: string; citations?: Citation[]; grounded?: boolean; intent?: string; rerank_confidence?: number; }
export interface Document { name: string; type: string; size_kb: number; uploaded: string; }
export interface HealthStatus { status: string; vectorstore_loaded: boolean; tenant: string; doc_count: number; }
export interface GapAnalysis { health_score: number; health_explanation: string; coverage_gaps: CoverageGap[]; underperforming_docs: UnderperformingDoc[]; stale_docs: StaleDoc[]; missing_common_docs: MissingDoc[]; }
export interface CoverageGap { topic: string; example_questions: string[]; suggested_document_title: string; }
export interface UnderperformingDoc { filename: string; reason: string; }
export interface StaleDoc { filename: string; days_since_upload: number; recommendation: string; }
export interface MissingDoc { title: string; why_needed: string; }
export interface ConflictResult { topic: string; description: string; source_1: string; source_2: string; excerpt_1: string; excerpt_2: string; severity: "high" | "medium" | "low"; }
