"use client";
import { useState, useEffect, useRef } from "react";
import { getDocuments, uploadDocument, triggerRebuild } from "@/lib/api";
import { Document } from "@/lib/types";

export default function DocumentsPage() {
  const [adminCode, setAdminCode] = useState("");
  const [authed, setAuthed] = useState(false);
  const [docs, setDocs] = useState<Document[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadDocs = async () => {
    setLoading(true);
    try { setDocs(await getDocuments()); }
    catch {}
    setLoading(false);
  };

  useEffect(() => { if (authed) loadDocs(); }, [authed]);

  const handleUpload = async (file: File) => {
    setStatus("Uploading...");
    try {
      const r = await uploadDocument(file, adminCode);
      setStatus(`Uploaded ${r.filename} — ${r.doc_count} docs, ${r.chunk_count} chunks indexed`);
      await loadDocs();
    } catch { setStatus("error:Upload failed. Check admin code."); }
  };

  const handleRebuild = async () => {
    setStatus("Rebuilding knowledge base...");
    try {
      const r = await triggerRebuild(adminCode);
      setStatus(`Rebuilt — ${r.doc_count} docs, ${r.chunk_count} chunks indexed`);
    } catch { setStatus("error:Rebuild failed."); }
  };

  const statusColor = status?.startsWith("error:") ? "#e05555" : status?.startsWith("Rebuilding") || status?.startsWith("Uploading") ? "#f0ad4e" : "var(--teal)";
  const statusText = status?.startsWith("error:") ? status.slice(6) : status;

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
          Document Workspace
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
      <h1 style={{ fontWeight: 500, fontSize: "1.5rem", marginBottom: "0.375rem", color: "var(--text-primary)" }}>Document Workspace</h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "2rem", fontSize: "0.9rem" }}>Manage the documents in your knowledge base</p>

      {status && (
        <div style={{
          padding: "0.625rem 1rem",
          background: "var(--surface)",
          border: `0.5px solid ${statusColor === "#e05555" ? "rgba(224,85,85,0.2)" : statusColor === "#f0ad4e" ? "rgba(240,173,78,0.2)" : "rgba(0,201,167,0.2)"}`,
          borderRadius: 8,
          marginBottom: "1.5rem",
          color: statusColor,
          fontSize: "0.875rem",
          fontFamily: "var(--font-mono)",
        }}>
          {statusText}
        </div>
      )}

      <div style={{ display: "flex", gap: "0.75rem", marginBottom: "2rem", flexWrap: "wrap" }}>
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleUpload(f); }}
          style={{
            border: "1.5px dashed rgba(0,201,167,0.25)",
            borderRadius: 12,
            padding: "1.25rem 2rem",
            cursor: "pointer",
            textAlign: "center",
            color: "var(--text-muted)",
            flex: 1,
            minWidth: 200,
            fontSize: "0.875rem",
            transition: "border-color 0.15s, color 0.15s",
          }}
          onMouseOver={e => {
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,201,167,0.5)";
            (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
          }}
          onMouseOut={e => {
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,201,167,0.25)";
            (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
          }}
        >
          Drop a PDF or DOCX here, or click to browse
          <input ref={fileRef} type="file" accept=".pdf,.docx" style={{ display: "none" }} onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
        </div>
        <button
          onClick={handleRebuild}
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
            transition: "all 0.15s",
          }}
          onMouseOver={e => {
            (e.currentTarget as HTMLElement).style.background = "rgba(0,201,167,0.07)";
          }}
          onMouseOut={e => {
            (e.currentTarget as HTMLElement).style.background = "transparent";
          }}
        >
          Rebuild Knowledge Base
        </button>
        <button
          onClick={loadDocs}
          style={{
            padding: "0.75rem 1.25rem",
            background: "transparent",
            color: "var(--text-muted)",
            border: "0.5px solid rgba(255,255,255,0.08)",
            borderRadius: 10,
            fontSize: "0.875rem",
            fontFamily: "var(--font-ui)",
            cursor: "pointer",
            transition: "all 0.15s",
          }}
          onMouseOver={e => {
            (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.15)";
          }}
          onMouseOut={e => {
            (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.08)";
          }}
        >
          Refresh
        </button>
      </div>

      {loading ? (
        <div style={{ color: "var(--text-muted)", fontSize: "0.875rem", fontFamily: "var(--font-mono)" }}>Loading documents...</div>
      ) : (
        <div style={{
          background: "var(--surface)",
          border: "0.5px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
          overflow: "hidden",
        }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "rgba(0,201,167,0.06)" }}>
                {["Document", "Type", "Size", "Uploaded"].map(h => (
                  <th key={h} style={{
                    textAlign: "left",
                    padding: "0.625rem 1rem",
                    color: "var(--text-secondary)",
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.72rem",
                    fontWeight: 500,
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {docs.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ color: "var(--text-muted)", padding: "1.5rem 1rem", fontSize: "0.875rem" }}>
                    No documents uploaded yet.
                  </td>
                </tr>
              ) : docs.map((doc, i) => (
                <tr
                  key={i}
                  style={{ borderBottom: "0.5px solid rgba(255,255,255,0.04)", transition: "background 0.1s" }}
                  onMouseOver={e => (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.02)"}
                  onMouseOut={e => (e.currentTarget as HTMLElement).style.background = "transparent"}
                >
                  <td style={{ padding: "0.625rem 1rem", fontSize: "0.875rem", color: "var(--text-primary)" }}>{doc.name}</td>
                  <td style={{ padding: "0.625rem 1rem", fontSize: "0.8rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{doc.type}</td>
                  <td style={{ padding: "0.625rem 1rem", fontSize: "0.8rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{doc.size_kb} KB</td>
                  <td style={{ padding: "0.625rem 1rem", fontSize: "0.8rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{doc.uploaded}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
