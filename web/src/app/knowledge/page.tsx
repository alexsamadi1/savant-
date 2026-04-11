"use client";
import { useState } from "react";
import { runGapAnalysis, runConflictDetection } from "@/lib/api";
import { GapAnalysis, ConflictResult } from "@/lib/types";

export default function KnowledgePage() {
  const [adminCode, setAdminCode] = useState("");
  const [authed, setAuthed] = useState(false);
  const [gapResult, setGapResult] = useState<GapAnalysis | null>(null);
  const [conflicts, setConflicts] = useState<ConflictResult[] | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleGapAnalysis = async () => {
    setLoading("gap"); setError(null);
    try { const r = await runGapAnalysis(adminCode); setGapResult(r); }
    catch { setError("Gap analysis failed. Check admin code."); }
    setLoading(null);
  };

  const handleConflicts = async () => {
    setLoading("conflicts"); setError(null);
    try { const r = await runConflictDetection(adminCode); setConflicts(r); }
    catch { setError("Conflict detection failed."); }
    setLoading(null);
  };

  const score = gapResult?.health_score ?? null;
  const scoreColor = score === null ? "#888" : score >= 70 ? "#00C9A7" : score >= 40 ? "#f0ad4e" : "#d9534f";

  if (!authed) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
      <div style={{ backgroundColor: "#0d1f1e", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 16, padding: "2rem", maxWidth: 360, width: "100%" }}>
        <h2 style={{ color: "#00C9A7", marginBottom: "1rem" }}>Admin Access</h2>
        <input type="password" value={adminCode} onChange={e => setAdminCode(e.target.value)} placeholder="Enter admin code"
          style={{ width: "100%", padding: "0.6rem", marginBottom: "1rem", backgroundColor: "#0a0a0f", color: "#e0e0e0", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 8, boxSizing: "border-box" }} />
        <button onClick={() => adminCode && setAuthed(true)} style={{ width: "100%", padding: "0.7rem", backgroundColor: "#00C9A7", color: "#fff", border: "none", borderRadius: 8, fontWeight: 600, cursor: "pointer" }}>
          Continue
        </button>
      </div>
    </div>
  );

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1rem" }}>
      <h1 style={{ marginBottom: "0.5rem" }}>Knowledge Map</h1>
      <p style={{ color: "#888", marginBottom: "2rem", fontSize: "0.9rem" }}>AI-powered audit of your organization's documentation coverage</p>

      {score !== null && (
        <div style={{ textAlign: "center", marginBottom: "2rem", padding: "1.5rem", backgroundColor: "#0d1f1e", borderRadius: 12 }}>
          <span style={{ fontSize: "3.5rem", fontWeight: 700, color: scoreColor }}>{score}</span>
          <span style={{ color: "#888" }}>/100</span>
          <p style={{ color: "#aaa", marginTop: "0.5rem" }}>{gapResult?.health_explanation}</p>
        </div>
      )}

      {error && <div style={{ color: "#ff6b6b", marginBottom: "1rem", padding: "0.75rem", backgroundColor: "rgba(255,107,107,0.1)", borderRadius: 8 }}>{error}</div>}

      <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem" }}>
        <button onClick={handleGapAnalysis} disabled={loading === "gap"}
          style={{ padding: "0.75rem 1.5rem", backgroundColor: "#00C9A7", color: "#fff", border: "none", borderRadius: 10, fontWeight: 600, cursor: "pointer", opacity: loading === "gap" ? 0.6 : 1 }}>
          {loading === "gap" ? "Analyzing..." : "Run Gap Analysis"}
        </button>
        <button onClick={handleConflicts} disabled={loading === "conflicts"}
          style={{ padding: "0.75rem 1.5rem", backgroundColor: "transparent", color: "#00C9A7", border: "1px solid #00C9A7", borderRadius: 10, fontWeight: 600, cursor: "pointer", opacity: loading === "conflicts" ? 0.6 : 1 }}>
          {loading === "conflicts" ? "Scanning..." : "Run Conflict Detection"}
        </button>
      </div>

      {gapResult?.coverage_gaps && gapResult.coverage_gaps.length > 0 && (
        <section style={{ marginBottom: "2rem" }}>
          <h2 style={{ color: "#f0ad4e", marginBottom: "1rem" }}>⚠️ Coverage Gaps ({gapResult.coverage_gaps.length})</h2>
          {gapResult.coverage_gaps.map((gap, i) => (
            <div key={i} style={{ backgroundColor: "#0d1f1e", borderLeft: "3px solid #f0ad4e", padding: "1rem", borderRadius: 8, marginBottom: "0.75rem" }}>
              <strong>{gap.topic}</strong>
              {gap.example_questions.map((q, j) => <div key={j} style={{ color: "#888", fontSize: "0.85rem", marginTop: "0.25rem" }}>• {q}</div>)}
              {gap.suggested_document_title && <div style={{ color: "#00C9A7", fontSize: "0.85rem", marginTop: "0.5rem" }}>📎 Suggested: {gap.suggested_document_title}</div>}
            </div>
          ))}
        </section>
      )}

      {conflicts !== null && (
        <section style={{ marginBottom: "2rem" }}>
          <h2 style={{ marginBottom: "1rem" }}>⚔️ Conflicts ({conflicts.length})</h2>
          {conflicts.length === 0 ? <div style={{ color: "#00C9A7" }}>✅ No conflicts detected.</div> : conflicts.map((c, i) => (
            <div key={i} style={{ backgroundColor: "#0d1f1e", borderLeft: `3px solid ${c.severity === "high" ? "#d9534f" : c.severity === "medium" ? "#f0ad4e" : "#00C9A7"}`, padding: "1rem", borderRadius: 8, marginBottom: "0.75rem" }}>
              <strong>{c.topic}</strong> — {c.description}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginTop: "0.5rem" }}>
                <div><div style={{ color: "#888", fontSize: "0.78rem" }}>{c.source_1}</div><div style={{ fontSize: "0.85rem", fontStyle: "italic" }}>"{c.excerpt_1}"</div></div>
                <div><div style={{ color: "#888", fontSize: "0.78rem" }}>{c.source_2}</div><div style={{ fontSize: "0.85rem", fontStyle: "italic" }}>"{c.excerpt_2}"</div></div>
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
