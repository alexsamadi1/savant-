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
  const scoreColor = score === null ? "var(--text-muted)" : score >= 70 ? "var(--teal)" : score >= 40 ? "#f0ad4e" : "#d9534f";

  if (!authed) return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: "2rem",
    }}>
      <div style={{
        background: "var(--surface)",
        border: "0.5px solid rgba(0,201,167,0.2)",
        borderRadius: 16,
        padding: "2rem 2.25rem",
        maxWidth: 400,
        width: "100%",
      }}>
        <p style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.68rem",
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          color: "var(--teal)",
          opacity: 0.6,
          marginBottom: "0.75rem",
        }}>
          Admin Required
        </p>
        <h2 style={{
          fontWeight: 500,
          fontSize: "1.375rem",
          lineHeight: 1.3,
          marginBottom: "1.5rem",
          color: "var(--text-primary)",
        }}>
          Knowledge Map Access
        </h2>
        <input
          type="password"
          value={adminCode}
          onChange={e => setAdminCode(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && adminCode) setAuthed(true); }}
          placeholder="Enter admin code"
          style={{
            width: "100%",
            padding: "0.625rem 0.875rem",
            background: "var(--surface-raised)",
            color: "var(--text-primary)",
            border: "0.5px solid rgba(0,201,167,0.2)",
            borderRadius: 9,
            fontSize: "0.9rem",
            fontFamily: "var(--font-ui)",
            outline: "none",
            marginBottom: "1.125rem",
            boxSizing: "border-box",
          }}
        />
        <button
          onClick={() => adminCode && setAuthed(true)}
          style={{
            width: "100%",
            padding: "0.75rem",
            background: "var(--teal)",
            color: "var(--bg)",
            border: "none",
            borderRadius: 10,
            fontWeight: 600,
            fontSize: "0.9375rem",
            fontFamily: "var(--font-ui)",
            cursor: "pointer",
            transition: "opacity 0.15s",
          }}
          onMouseOver={e => (e.currentTarget as HTMLElement).style.opacity = "0.85"}
          onMouseOut={e => (e.currentTarget as HTMLElement).style.opacity = "1"}
        >
          Continue
        </button>
      </div>
    </div>
  );

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: "2rem 1.5rem", fontFamily: "var(--font-ui)" }}>
      <p style={{
        fontFamily: "var(--font-mono)",
        fontSize: "0.68rem",
        letterSpacing: "0.15em",
        textTransform: "uppercase",
        color: "var(--teal)",
        opacity: 0.6,
        marginBottom: "0.5rem",
      }}>
        Admin
      </p>
      <h1 style={{ fontWeight: 500, fontSize: "1.5rem", marginBottom: "0.375rem", color: "var(--text-primary)" }}>Knowledge Map</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "2rem", fontSize: "0.9rem" }}>AI-powered audit of your organization&apos;s documentation coverage</p>

      {score !== null && (
        <div style={{
          textAlign: "center",
          marginBottom: "2rem",
          padding: "1.5rem",
          background: "var(--surface)",
          border: "0.5px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
        }}>
          <span style={{ fontSize: "3.5rem", fontWeight: 500, color: scoreColor }}>{score}</span>
          <span style={{ color: "var(--text-muted)" }}>/100</span>
          <p style={{ color: "var(--text-secondary)", marginTop: "0.5rem", fontSize: "0.9rem" }}>{gapResult?.health_explanation}</p>
        </div>
      )}

      {error && (
        <div style={{
          color: "#e05555",
          marginBottom: "1rem",
          padding: "0.75rem 1rem",
          background: "rgba(224,85,85,0.08)",
          border: "0.5px solid rgba(224,85,85,0.2)",
          borderRadius: 8,
          fontSize: "0.875rem",
        }}>
          {error}
        </div>
      )}

      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "2rem" }}>
        <button
          onClick={handleGapAnalysis}
          disabled={loading === "gap"}
          style={{
            padding: "0.75rem 1.5rem",
            background: "var(--teal)",
            color: "var(--bg)",
            border: "none",
            borderRadius: 10,
            fontWeight: 600,
            fontSize: "0.875rem",
            fontFamily: "var(--font-ui)",
            cursor: "pointer",
            opacity: loading === "gap" ? 0.6 : 1,
            transition: "opacity 0.15s",
          }}
        >
          {loading === "gap" ? "Analyzing..." : "Run Gap Analysis"}
        </button>
        <button
          onClick={handleConflicts}
          disabled={loading === "conflicts"}
          style={{
            padding: "0.75rem 1.5rem",
            background: "transparent",
            color: "var(--teal)",
            border: "0.5px solid rgba(0,201,167,0.3)",
            borderRadius: 10,
            fontWeight: 600,
            fontSize: "0.875rem",
            fontFamily: "var(--font-ui)",
            cursor: "pointer",
            opacity: loading === "conflicts" ? 0.6 : 1,
            transition: "opacity 0.15s",
          }}
        >
          {loading === "conflicts" ? "Scanning..." : "Run Conflict Detection"}
        </button>
      </div>

      {gapResult?.coverage_gaps && gapResult.coverage_gaps.length > 0 && (
        <section style={{ marginBottom: "2rem" }}>
          <h2 style={{ fontWeight: 500, fontSize: "1.1rem", color: "#f0ad4e", marginBottom: "1rem" }}>Coverage Gaps ({gapResult.coverage_gaps.length})</h2>
          {gapResult.coverage_gaps.map((gap, i) => (
            <div key={i} style={{
              background: "var(--surface)",
              border: "0.5px solid rgba(255,255,255,0.06)",
              borderLeft: "2.5px solid #f0ad4e",
              padding: "1rem 1.25rem",
              borderRadius: "0 12px 12px 0",
              marginBottom: "0.75rem",
            }}>
              <strong style={{ fontWeight: 500, color: "var(--text-primary)" }}>{gap.topic}</strong>
              {gap.example_questions.map((q, j) => (
                <div key={j} style={{ color: "var(--text-secondary)", fontSize: "0.85rem", marginTop: "0.25rem" }}>• {q}</div>
              ))}
              {gap.suggested_document_title && (
                <div style={{ color: "var(--teal)", fontSize: "0.85rem", marginTop: "0.5rem", fontFamily: "var(--font-mono)", opacity: 0.8 }}>
                  Suggested: {gap.suggested_document_title}
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      {conflicts !== null && (
        <section style={{ marginBottom: "2rem" }}>
          <h2 style={{ fontWeight: 500, fontSize: "1.1rem", color: "var(--text-primary)", marginBottom: "1rem" }}>Conflicts ({conflicts.length})</h2>
          {conflicts.length === 0 ? (
            <div style={{ color: "var(--teal)", fontSize: "0.9rem" }}>No conflicts detected.</div>
          ) : conflicts.map((c, i) => (
            <div key={i} style={{
              background: "var(--surface)",
              border: "0.5px solid rgba(255,255,255,0.06)",
              borderLeft: `2.5px solid ${c.severity === "high" ? "#d9534f" : c.severity === "medium" ? "#f0ad4e" : "var(--teal)"}`,
              padding: "1rem 1.25rem",
              borderRadius: "0 12px 12px 0",
              marginBottom: "0.75rem",
            }}>
              <strong style={{ fontWeight: 500, color: "var(--text-primary)" }}>{c.topic}</strong>
              <span style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}> — {c.description}</span>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginTop: "0.625rem" }}>
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)", fontSize: "0.7rem", marginBottom: "0.2rem" }}>{c.source_1}</div>
                  <div style={{ fontSize: "0.85rem", fontStyle: "italic", color: "var(--text-secondary)" }}>&ldquo;{c.excerpt_1}&rdquo;</div>
                </div>
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-muted)", fontSize: "0.7rem", marginBottom: "0.2rem" }}>{c.source_2}</div>
                  <div style={{ fontSize: "0.85rem", fontStyle: "italic", color: "var(--text-secondary)" }}>&ldquo;{c.excerpt_2}&rdquo;</div>
                </div>
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
