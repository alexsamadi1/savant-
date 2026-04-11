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
      setStatus(`✅ Uploaded ${r.filename} — ${r.doc_count} docs, ${r.chunk_count} chunks indexed`);
      await loadDocs();
    } catch { setStatus("❌ Upload failed. Check admin code."); }
  };

  const handleRebuild = async () => {
    setStatus("Rebuilding knowledge base...");
    try {
      const r = await triggerRebuild(adminCode);
      setStatus(`✅ Rebuilt — ${r.doc_count} docs, ${r.chunk_count} chunks indexed`);
    } catch { setStatus("❌ Rebuild failed."); }
  };

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
      <h1 style={{ marginBottom: "0.5rem" }}>Document Workspace</h1>
      <p style={{ color: "#888", marginBottom: "2rem", fontSize: "0.9rem" }}>Manage the documents in your knowledge base</p>

      {status && <div style={{ padding: "0.75rem 1rem", backgroundColor: "#0d1f1e", borderRadius: 8, marginBottom: "1.5rem", color: status.startsWith("✅") ? "#00C9A7" : status.startsWith("❌") ? "#ff6b6b" : "#e0e0e0", fontSize: "0.9rem" }}>{status}</div>}

      <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem", flexWrap: "wrap" }}>
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleUpload(f); }}
          style={{ border: "2px dashed rgba(0,201,167,0.4)", borderRadius: 12, padding: "1.5rem 2rem", cursor: "pointer", textAlign: "center", color: "#888", flex: 1, minWidth: 200 }}>
          📎 Drop a PDF or DOCX here, or click to browse
          <input ref={fileRef} type="file" accept=".pdf,.docx" style={{ display: "none" }} onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f); }} />
        </div>
        <button onClick={handleRebuild} style={{ padding: "0.75rem 1.5rem", backgroundColor: "transparent", color: "#00C9A7", border: "1px solid #00C9A7", borderRadius: 10, fontWeight: 600, cursor: "pointer" }}>
          🔄 Rebuild Knowledge Base
        </button>
        <button onClick={loadDocs} style={{ padding: "0.75rem 1.25rem", backgroundColor: "transparent", color: "#666", border: "1px solid #333", borderRadius: 10, cursor: "pointer" }}>
          ↻ Refresh
        </button>
      </div>

      {loading ? <div style={{ color: "#888" }}>Loading documents...</div> : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(0,201,167,0.2)" }}>
              {["Document", "Type", "Size", "Uploaded"].map(h => (
                <th key={h} style={{ textAlign: "left", padding: "0.6rem 0.75rem", color: "#888", fontSize: "0.82rem", fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {docs.length === 0 ? (
              <tr><td colSpan={4} style={{ color: "#666", padding: "1.5rem 0.75rem", fontSize: "0.9rem" }}>No documents uploaded yet.</td></tr>
            ) : docs.map((doc, i) => (
              <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                <td style={{ padding: "0.6rem 0.75rem", fontSize: "0.9rem" }}>{doc.name}</td>
                <td style={{ padding: "0.6rem 0.75rem", fontSize: "0.82rem", color: "#888" }}>{doc.type}</td>
                <td style={{ padding: "0.6rem 0.75rem", fontSize: "0.82rem", color: "#888" }}>{doc.size_kb} KB</td>
                <td style={{ padding: "0.6rem 0.75rem", fontSize: "0.82rem", color: "#888" }}>{doc.uploaded}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
