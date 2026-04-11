import { HealthStatus, Document, GapAnalysis, ConflictResult, UserProfile } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const TENANT = process.env.NEXT_PUBLIC_TENANT || "demo";

export async function getHealth(): Promise<HealthStatus> {
  const r = await fetch(`${BASE}/health?tenant=${TENANT}`);
  return r.json();
}
export async function getDocuments(): Promise<Document[]> {
  const r = await fetch(`${BASE}/documents?tenant=${TENANT}`);
  return r.json();
}
export async function uploadDocument(file: File, adminCode: string) {
  const form = new FormData();
  form.append("file", file);
  form.append("tenant", TENANT);
  const r = await fetch(`${BASE}/upload`, { method: "POST", headers: { "X-Admin-Code": adminCode }, body: form });
  return r.json();
}
export async function triggerRebuild(adminCode: string) {
  const r = await fetch(`${BASE}/rebuild`, { method: "POST", headers: { "Content-Type": "application/json", "X-Admin-Code": adminCode }, body: JSON.stringify({ tenant: TENANT }) });
  return r.json();
}
export async function runGapAnalysis(adminCode: string): Promise<GapAnalysis> {
  const r = await fetch(`${BASE}/admin/gap-analysis`, { method: "POST", headers: { "Content-Type": "application/json", "X-Admin-Code": adminCode }, body: JSON.stringify({ tenant: TENANT }) });
  return r.json();
}
export async function runConflictDetection(adminCode: string): Promise<ConflictResult[]> {
  const r = await fetch(`${BASE}/admin/conflicts`, { method: "POST", headers: { "Content-Type": "application/json", "X-Admin-Code": adminCode }, body: JSON.stringify({ tenant: TENANT }) });
  return r.json();
}
export function getStreamUrl() { return `${BASE}/query/stream`; }
export function buildStreamBody(question: string, profile: UserProfile, model: string) {
  return JSON.stringify({ question, profile, model, tenant: TENANT, include_follow_ups: true });
}
