"use client";
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useStream } from "@/lib/useStream";
import { UserProfile, Citation } from "@/lib/types";

export default function QueryPage() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [onboarding, setOnboarding] = useState(false);
  const [role, setRole] = useState("Manager");
  const [tenure, setTenure] = useState("6+ Months");
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<Array<{ q: string; a: string; citations: Citation[]; grounded: boolean | null; confidence: number | null; intent: string | null; latency: number | null }>>([]);
  const { tokens, citations, grounded, confidence, intent, status, latency, ask, reset } = useStream();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const saved = localStorage.getItem("savant_profile");
    if (saved) { try { setProfile(JSON.parse(saved)); } catch {} }
    else setOnboarding(true);
  }, []);

  useEffect(() => {
    if (status === "done" && tokens) {
      setHistory(h => [...h, { q: history.length > 0 && history[history.length-1]?.q === question ? question : question, a: tokens, citations, grounded, confidence, intent, latency }]);
      reset();
      setQuestion("");
    }
  }, [status]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [tokens, history]);

  const handleSubmit = (q: string) => {
    if (!q.trim() || !profile || status === "searching" || status === "streaming") return;
    ask(q, profile);
  };

  const saveProfile = () => {
    const p = { role, tenure };
    localStorage.setItem("savant_profile", JSON.stringify(p));
    setProfile(p);
    setOnboarding(false);
  };

  if (onboarding) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", padding: "2rem" }}>
      <div style={{ backgroundColor: "#0d1f1e", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 16, padding: "2rem", maxWidth: 420, width: "100%" }}>
        <h2 style={{ color: "#00C9A7", marginBottom: "1.5rem", textAlign: "center" }}>Welcome to Savant</h2>
        <label style={{ display: "block", marginBottom: "0.5rem", color: "#888", fontSize: "0.85rem" }}>Your role</label>
        <select value={role} onChange={e => setRole(e.target.value)} style={{ width: "100%", padding: "0.6rem", marginBottom: "1rem", backgroundColor: "#0a0a0f", color: "#e0e0e0", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 8 }}>
          {["Manager", "Individual Contributor", "New Hire", "Other"].map(r => <option key={r}>{r}</option>)}
        </select>
        <label style={{ display: "block", marginBottom: "0.5rem", color: "#888", fontSize: "0.85rem" }}>Time at company</label>
        <select value={tenure} onChange={e => setTenure(e.target.value)} style={{ width: "100%", padding: "0.6rem", marginBottom: "1.5rem", backgroundColor: "#0a0a0f", color: "#e0e0e0", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 8 }}>
          {["New Hire (0–30 days)", "1–6 Months", "6+ Months", "2+ Years"].map(t => <option key={t}>{t}</option>)}
        </select>
        <button onClick={saveProfile} style={{ width: "100%", padding: "0.75rem", backgroundColor: "#00C9A7", color: "#fff", border: "none", borderRadius: 10, fontWeight: 600, cursor: "pointer", fontSize: "1rem" }}>
          Continue →
        </button>
      </div>
    </div>
  );

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "2rem 1rem" }}>
      <h1 style={{ textAlign: "center", marginBottom: "0.5rem", fontSize: "1.75rem" }}>Savant Assistant</h1>
      <p style={{ textAlign: "center", color: "#888", marginBottom: "2rem", fontSize: "0.9rem" }}>Ask anything about your organization's knowledge base</p>

      {history.map((item, i) => (
        <div key={i} style={{ marginBottom: "2rem" }}>
          <div style={{ backgroundColor: "#00C9A7", color: "#fff", padding: "0.75rem 1rem", borderRadius: 18, marginBottom: "0.5rem", display: "inline-block", maxWidth: "80%" }}>
            {item.q}
          </div>
          <div style={{ backgroundColor: "#0d1f1e", borderLeft: "3px solid #00C9A7", padding: "1rem 1.25rem", borderRadius: 12, color: "#e0e0e0", lineHeight: 1.7 }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{item.a}</ReactMarkdown>
            {item.citations.length > 0 && (
              <div style={{ marginTop: "0.75rem", display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
                {item.citations.map((c, j) => (
                  <span key={j} style={{ fontSize: "0.8rem", backgroundColor: "rgba(0,201,167,0.1)", color: "#00C9A7", border: "1px solid rgba(0,201,167,0.4)", borderRadius: 6, padding: "3px 10px" }}>
                    📄 {c.source}{c.section ? ` — ${c.section}` : ""}
                  </span>
                ))}
              </div>
            )}
            <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.75rem", fontSize: "0.78rem", color: "#666" }}>
              {item.grounded === true && <span style={{ color: "#00C9A7" }}>✅ Verified</span>}
              {item.grounded === false && <span style={{ color: "#f0ad4e" }}>🟡 Unverified</span>}
              {item.intent && <span>{item.intent === "synthesis" ? "Deep analysis" : "Quick answer"}</span>}
              {item.latency && <span>{item.latency.toFixed(1)}s</span>}
            </div>
          </div>
        </div>
      ))}

      {(status === "searching" || status === "streaming") && (
        <div style={{ marginBottom: "2rem" }}>
          {status === "searching" && (
            <div style={{ color: "#888", fontSize: "0.85rem", marginBottom: "0.5rem", animation: "pulse 1.5s infinite" }}>
              🔍 Searching your documents...
            </div>
          )}
          {tokens && (
            <div style={{ backgroundColor: "#0d1f1e", borderLeft: "3px solid #00C9A7", padding: "1rem 1.25rem", borderRadius: 12, color: "#e0e0e0", lineHeight: 1.7 }}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{tokens}</ReactMarkdown>
              <span style={{ color: "#00C9A7", animation: "blink 1s infinite" }}>▌</span>
            </div>
          )}
        </div>
      )}

      <div ref={bottomRef} />

      <div style={{ position: "sticky", bottom: "1rem", backgroundColor: "#0a0a0f", paddingTop: "1rem" }}>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(question); } }}
            placeholder="Ask anything about your organization..."
            style={{ flex: 1, padding: "0.85rem 1rem", backgroundColor: "#0d1f1e", color: "#e0e0e0", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 12, fontSize: "0.95rem", outline: "none" }}
          />
          <button
            onClick={() => handleSubmit(question)}
            disabled={!question.trim() || status === "searching" || status === "streaming"}
            style={{ padding: "0.85rem 1.5rem", backgroundColor: "#00C9A7", color: "#fff", border: "none", borderRadius: 12, fontWeight: 600, cursor: "pointer", opacity: (!question.trim() || status !== "idle") ? 0.5 : 1 }}>
            Ask
          </button>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.5rem", flexWrap: "wrap" }}>
          {["What is the PTO policy?", "How do I submit expenses?", "What are the security requirements?"].map(q => (
            <button key={q} onClick={() => { setQuestion(q); handleSubmit(q); }} style={{ fontSize: "0.78rem", padding: "4px 10px", backgroundColor: "transparent", color: "#00C9A7", border: "1px solid rgba(0,201,167,0.3)", borderRadius: 20, cursor: "pointer" }}>
              {q}
            </button>
          ))}
          {history.length > 0 && (
            <button onClick={() => { setHistory([]); }} style={{ fontSize: "0.78rem", padding: "4px 10px", backgroundColor: "transparent", color: "#666", border: "1px solid #333", borderRadius: 20, cursor: "pointer", marginLeft: "auto" }}>
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
