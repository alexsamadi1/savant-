"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { useStream } from "@/lib/useStream";
import { UserProfile, Citation } from "@/lib/types";

const SAMPLE_QUESTIONS = [
  "What are the technologies used across projects?",
  "Which programs are behind schedule?",
  "What is the PTO policy?",
];

interface HistoryItem {
  q: string;
  answer: string;
  citations: Citation[];
  grounded: boolean | null;
  intent: string | null;
  confidence: number | null;
  latency: number | null;
  followUps: string[];
}

export default function QueryPage() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [role, setRole] = useState("Manager");
  const [tenure, setTenure] = useState("6+ Months");
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activityStage, setActivityStage] = useState(0);
  const [docCount, setDocCount] = useState<number | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [collapsedItems, setCollapsedItems] = useState<Set<number>>(new Set());
  const [currentFollowUps, setCurrentFollowUps] = useState<string[]>([]);
  const activityRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const { tokens, citations, grounded, intent, confidence, status, latency, followUps, ask, reset } = useStream();

  const STAGES = [
    "Rewriting query...",
    "Scanning document sections...",
    "Reranking with cross-encoder...",
    "Generating answer...",
  ];

  useEffect(() => {
    const saved = localStorage.getItem("savant_profile");
    if (saved) {
      try { setProfile(JSON.parse(saved)); } catch {}
    } else {
      setShowOnboarding(true);
    }
  }, []);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/health?tenant=${process.env.NEXT_PUBLIC_TENANT || 'demo'}`)
      .then(r => r.json())
      .then(data => {
        setDocCount(data.doc_count ?? null);
        setLastUpdated(data.last_updated ?? null);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (status === "searching") {
      setActivityStage(0);
      let s = 0;
      activityRef.current = setInterval(() => {
        s++;
        if (s < STAGES.length) setActivityStage(s);
        else if (activityRef.current) clearInterval(activityRef.current);
      }, 600);
    } else {
      if (activityRef.current) clearInterval(activityRef.current);
    }
  }, [status]);

  useEffect(() => {
    if (status === "done" && tokens) {
      setCollapsedItems(prev => {
        const next = new Set(prev);
        if (history.length > 0) next.add(history.length - 1);
        return next;
      });
      setHistory(h => [...h, {
        q: history.length > 0 ? history[history.length - 1].q : question,
        answer: tokens,
        citations: citations || [],
        grounded: grounded,
        intent: intent,
        confidence: confidence,
        latency: latency,
        followUps: followUps || [],
      }]);
      setCurrentFollowUps(followUps || []);
      reset();
    }
  }, [status]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [tokens, history, status]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setHistory([]);
        setCollapsedItems(new Set());
        setCurrentFollowUps([]);
        reset();
        setQuestion("");
        setTimeout(() => inputRef.current?.focus(), 50);
      }
      if (e.key === "Escape") {
        inputRef.current?.blur();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const handleAsk = useCallback((q: string) => {
    if (!q.trim() || !profile || status === "searching" || status === "streaming") return;
    setCurrentFollowUps([]);
    ask(q, profile);
    setQuestion("");
  }, [profile, status, ask]);

  const saveProfile = () => {
    const p = { role, tenure };
    localStorage.setItem("savant_profile", JSON.stringify(p));
    setProfile(p);
    setShowOnboarding(false);
  };

  const hasContent = history.length > 0 || status === "searching" || status === "streaming";

  if (showOnboarding) return <OnboardingModal role={role} tenure={tenure} setRole={setRole} setTenure={setTenure} onSave={saveProfile} />;

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: "0 1.5rem", display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <Topbar profile={profile} />

      {!hasContent && <Hero docCount={docCount} lastUpdated={lastUpdated} />}

      <div style={{ flex: 1, paddingBottom: "1rem" }}>
        {history.map((item, i) => {
          const isLast = i === history.length - 1;
          const isCollapsed = collapsedItems.has(i);
          return (
            <div key={i} style={{ marginBottom: isLast ? "1.75rem" : "0.5rem" }}>
              <div
                onClick={() => {
                  setCollapsedItems(prev => {
                    const next = new Set(prev);
                    if (next.has(i)) next.delete(i);
                    else next.add(i);
                    return next;
                  });
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.625rem",
                  cursor: "pointer",
                  padding: "0.3rem 0",
                  marginBottom: isCollapsed ? 0 : "0.5rem",
                }}
              >
                <svg
                  viewBox="0 0 24 24"
                  width={12}
                  height={12}
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  style={{
                    color: "var(--text-muted)",
                    flexShrink: 0,
                    transform: isCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
                    transition: "transform 0.15s",
                  }}
                >
                  <polyline points="6 9 12 15 18 9"/>
                </svg>
                <span style={{
                  fontFamily: "var(--font-ui)",
                  fontSize: "0.85rem",
                  color: isCollapsed ? "var(--text-muted)" : "var(--text-secondary)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  maxWidth: "600px",
                }}>
                  {item.q}
                </span>
              </div>
              {!isCollapsed && (
                <>
                  <QuestionBubble text={item.q} />
                  <AnswerCard
                    answer={item.answer}
                    citations={item.citations}
                    grounded={item.grounded}
                    intent={item.intent}
                    confidence={item.confidence}
                    latency={item.latency}
                  />
                  <FollowUps questions={item.followUps} onAsk={handleAsk} />
                </>
              )}
            </div>
          );
        })}

        {status === "searching" && (
          <ActivityCard stage={activityStage} stages={STAGES} />
        )}

        {status === "streaming" && tokens && (
          <div style={{ marginBottom: "1rem" }}>
            <div style={{
              background: "var(--surface)",
              borderLeft: "2.5px solid var(--teal)",
              borderRadius: "0 12px 12px 0",
              padding: "1rem 1.25rem",
              fontFamily: "var(--font-answer)",
              fontSize: "1rem",
              lineHeight: 1.8,
              color: "var(--text-primary)",
            }}>
              <MarkdownBody text={tokens} />
              <span style={{
                display: "inline-block",
                width: 2,
                height: "1em",
                background: "var(--teal)",
                marginLeft: 2,
                verticalAlign: "middle",
                animation: "blink 1s step-end infinite",
              }} />
            </div>
          </div>
        )}

        {status === "idle" && currentFollowUps.length > 0 && history.length > 0 && (
          <FollowUps questions={currentFollowUps} onAsk={handleAsk} />
        )}

        <div ref={bottomRef} />
      </div>

      <div style={{
        position: "sticky",
        bottom: 0,
        paddingTop: "0.75rem",
        paddingBottom: "1.25rem",
        background: "linear-gradient(to top, var(--bg) 70%, transparent)",
      }}>
        <div style={{ position: "relative" }}>
          <textarea
            ref={inputRef}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleAsk(question);
              }
            }}
            placeholder="Ask anything about your organization..."
            rows={1}
            style={{
              width: "100%",
              background: "var(--surface)",
              border: "0.5px solid rgba(0,201,167,0.2)",
              borderRadius: 14,
              padding: "0.875rem 3.25rem 0.875rem 1.125rem",
              color: "var(--text-primary)",
              fontFamily: "var(--font-ui)",
              fontSize: "0.9375rem",
              lineHeight: 1.5,
              resize: "none",
              outline: "none",
              transition: "border-color 0.15s, box-shadow 0.15s",
              minHeight: 52,
            }}
            onFocus={e => {
              e.currentTarget.style.borderColor = "rgba(0,201,167,0.45)";
              e.currentTarget.style.boxShadow = "0 0 0 3px rgba(0,201,167,0.06)";
            }}
            onBlur={e => {
              e.currentTarget.style.borderColor = "rgba(0,201,167,0.2)";
              e.currentTarget.style.boxShadow = "none";
            }}
          />
          <button
            onClick={() => handleAsk(question)}
            disabled={!question.trim() || status === "searching" || status === "streaming"}
            style={{
              position: "absolute",
              right: 10,
              top: "50%",
              transform: "translateY(-50%)",
              width: 32,
              height: 32,
              background: question.trim() ? "var(--teal)" : "transparent",
              border: question.trim() ? "none" : "0.5px solid rgba(0,201,167,0.25)",
              borderRadius: 9,
              cursor: question.trim() ? "pointer" : "not-allowed",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.15s",
              opacity: (status === "searching" || status === "streaming") ? 0.4 : 1,
            }}
          >
            <svg viewBox="0 0 24 24" width={13} height={13} fill={question.trim() ? "white" : "var(--teal)"}>
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>

        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginTop: "0.4rem",
          paddingLeft: "0.25rem",
        }}>
          <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
            {!hasContent && SAMPLE_QUESTIONS.map(q => (
              <button
                key={q}
                onClick={() => handleAsk(q)}
                style={{
                  fontSize: "0.78rem",
                  color: "var(--text-muted)",
                  background: "transparent",
                  border: "0.5px solid rgba(255,255,255,0.07)",
                  borderRadius: 20,
                  padding: "4px 12px",
                  cursor: "pointer",
                  fontFamily: "var(--font-ui)",
                  transition: "all 0.15s",
                }}
                onMouseOver={e => {
                  (e.currentTarget as HTMLElement).style.color = "var(--teal)";
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,201,167,0.3)";
                }}
                onMouseOut={e => {
                  (e.currentTarget as HTMLElement).style.color = "var(--text-muted)";
                  (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.07)";
                }}
              >
                {q}
              </button>
            ))}
          </div>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-muted)" }}>
            / to focus · Enter to send · ⌘K to clear
          </span>
        </div>
      </div>

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes settleIn { 0%{opacity:0.7} 50%{opacity:0.92} 100%{opacity:1} }
      `}</style>
    </div>
  );
}

function FollowUps({ questions, onAsk }: { questions: string[]; onAsk: (q: string) => void }) {
  if (!questions || questions.length === 0) return null;
  return (
    <div style={{ marginTop: "0.75rem", display: "flex", flexWrap: "wrap", gap: "0.4rem" }}>
      {questions.map((q, i) => (
        <button
          key={i}
          onClick={() => onAsk(q)}
          style={{
            fontSize: "0.8rem",
            color: "var(--text-secondary)",
            background: "transparent",
            border: "0.5px solid rgba(255,255,255,0.1)",
            borderRadius: 20,
            padding: "5px 14px",
            cursor: "pointer",
            fontFamily: "var(--font-ui)",
            transition: "all 0.15s",
            textAlign: "left",
          }}
          onMouseOver={e => {
            (e.currentTarget as HTMLElement).style.color = "var(--teal)";
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,201,167,0.3)";
            (e.currentTarget as HTMLElement).style.background = "rgba(0,201,167,0.05)";
          }}
          onMouseOut={e => {
            (e.currentTarget as HTMLElement).style.color = "var(--text-secondary)";
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.1)";
            (e.currentTarget as HTMLElement).style.background = "transparent";
          }}
        >
          ↳ {q}
        </button>
      ))}
    </div>
  );
}

function Topbar({ profile }: { profile: UserProfile | null }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 10,
      padding: "0.875rem 0",
      borderBottom: "0.5px solid rgba(0,201,167,0.08)",
      marginBottom: "0.5rem",
    }}>
      <span style={{
        fontFamily: "var(--font-mono)",
        fontSize: "0.7rem",
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: "var(--teal)",
        opacity: 0.7,
      }}>
        Savant
      </span>
      <span style={{
        fontFamily: "var(--font-mono)",
        fontSize: "0.68rem",
        color: "rgba(0,201,167,0.5)",
        background: "rgba(0,201,167,0.07)",
        border: "0.5px solid rgba(0,201,167,0.15)",
        padding: "2px 8px",
        borderRadius: 4,
      }}>
        {process.env.NEXT_PUBLIC_TENANT || "demo"}
      </span>
      <div style={{ flex: 1 }} />
      {profile && (
        <span style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.68rem",
          color: "var(--text-muted)",
        }}>
          {profile.role}
        </span>
      )}
    </div>
  );
}

function Hero({ docCount, lastUpdated }: { docCount: number | null; lastUpdated: string | null }) {
  return (
    <div style={{
      textAlign: "center",
      padding: "3rem 0 2rem",
      animation: "fadeUp 0.4s ease",
    }}>
      <p style={{
        fontFamily: "var(--font-mono)",
        fontSize: "0.68rem",
        letterSpacing: "0.2em",
        textTransform: "uppercase",
        color: "var(--teal)",
        opacity: 0.6,
        marginBottom: "0.875rem",
      }}>
        Organizational Intelligence
      </p>
      <h1 style={{
        fontFamily: "var(--font-ui)",
        fontWeight: 500,
        fontSize: "2rem",
        lineHeight: 1.2,
        color: "var(--text-primary)",
        marginBottom: "0.625rem",
        letterSpacing: "-0.02em",
      }}>
        Ask anything about your<br />knowledge base
      </h1>
      <p style={{
        fontSize: "0.9rem",
        color: "var(--text-secondary)",
        lineHeight: 1.6,
      }}>
        Cited, grounded, and auditable — built for GovCon
      </p>
      {docCount !== null && (
        <div style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "0.75rem",
          marginTop: "1.25rem",
          padding: "0.4rem 1rem",
          background: "rgba(255,255,255,0.03)",
          border: "0.5px solid rgba(255,255,255,0.08)",
          borderRadius: 20,
          fontFamily: "var(--font-mono)",
          fontSize: "0.68rem",
          color: "var(--text-muted)",
        }}>
          <span style={{ color: "var(--teal)", opacity: 0.8 }}>{docCount} document{docCount !== 1 ? "s" : ""}</span>
          {lastUpdated && (
            <>
              <span style={{ opacity: 0.3 }}>·</span>
              <span>updated {lastUpdated}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function QuestionBubble({ text }: { text: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "0.5rem" }}>
      <div style={{
        maxWidth: "72%",
        background: "rgba(0,201,167,0.12)",
        border: "0.5px solid rgba(0,201,167,0.2)",
        borderRadius: "16px 16px 4px 16px",
        padding: "0.625rem 1rem",
        fontSize: "0.9rem",
        color: "rgba(0,201,167,0.9)",
        lineHeight: 1.5,
        fontWeight: 400,
      }}>
        {text}
      </div>
    </div>
  );
}

function ActivityCard({ stage, stages }: { stage: number; stages: string[] }) {
  const progress = Math.round(((stage + 1) / stages.length) * 100);
  return (
    <div style={{
      background: "var(--surface)",
      border: "0.5px solid rgba(0,201,167,0.15)",
      borderRadius: 12,
      padding: "0.875rem 1.125rem",
      marginBottom: "0.75rem",
      animation: "fadeUp 0.25s ease",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.625rem", marginBottom: "0.625rem" }}>
        <div style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: "var(--teal)",
          animation: "pulse 1.5s ease infinite",
          flexShrink: 0,
        }} />
        <span style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.75rem",
          color: "var(--text-secondary)",
        }}>
          {stages[stage]}
        </span>
      </div>
      <div style={{
        height: 1.5,
        background: "rgba(255,255,255,0.06)",
        borderRadius: 1,
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${progress}%`,
          background: "var(--teal)",
          borderRadius: 1,
          transition: "width 0.5s ease",
          opacity: 0.7,
        }} />
      </div>
    </div>
  );
}

function AnswerCard({ answer, citations, grounded, intent, confidence, latency }: {
  answer: string;
  citations: Citation[];
  grounded: boolean | null;
  intent: string | null;
  confidence: number | null;
  latency: number | null;
}) {
  const [copied, setCopied] = useState(false);
  const confLevel = confidence === null ? 0 : confidence > 5 ? 3 : confidence > 2 ? 2 : 1;
  const isVerified = grounded === true;
  const confColor = isVerified ? "var(--teal)" : "#f0ad4e";
  const borderColor = grounded === false ? "#f0ad4e" : "var(--teal)";

  return (
    <div style={{ position: "relative", animation: "settleIn 0.25s ease" }}>
      <button
        onClick={() => {
          navigator.clipboard.writeText(answer.replace(/<[^>]+>/g, '').replace(/\*\*/g, ''));
          setCopied(true);
          setTimeout(() => setCopied(false), 2000);
        }}
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          background: "transparent",
          border: "0.5px solid rgba(255,255,255,0.1)",
          borderRadius: 6,
          padding: "4px 8px",
          cursor: "pointer",
          fontFamily: "var(--font-mono)",
          fontSize: "0.65rem",
          color: copied ? "var(--teal)" : "var(--text-muted)",
          transition: "all 0.15s",
          zIndex: 10,
        }}
        className="copy-btn"
      >
        {copied ? "copied" : "copy"}
      </button>
      <div style={{
        background: "var(--surface)",
        border: "0.5px solid rgba(255,255,255,0.06)",
        borderLeft: `2.5px solid ${borderColor}`,
        borderRadius: "0 12px 12px 0",
        animation: "fadeUp 0.3s ease",
      }}>
        <div style={{
          padding: "1.125rem 1.375rem",
          fontFamily: "var(--font-answer)",
          fontSize: "1rem",
          lineHeight: 1.8,
          color: "var(--text-primary)",
        }}>
          {grounded === false && (
            <div style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              background: "rgba(240,173,78,0.08)",
              border: "0.5px solid rgba(240,173,78,0.25)",
              borderRadius: 8,
              padding: "0.5rem 0.875rem",
              marginBottom: "0.875rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.72rem",
              color: "#f0ad4e",
            }}>
              <svg viewBox="0 0 24 24" width={12} height={12} fill="none" stroke="currentColor" strokeWidth="2" style={{flexShrink:0}}>
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
              </svg>
              Unverified — answer could not be fully confirmed against your documents. Validate before acting.
            </div>
          )}
          <MarkdownBody text={answer} />
        </div>

        {citations && citations.length > 0 && (
          <div style={{
            padding: "0.625rem 1.375rem",
            borderTop: "0.5px solid rgba(255,255,255,0.05)",
            display: "flex",
            flexWrap: "wrap",
            gap: "0.4rem",
          }}>
            {citations.map((c, i) => (
              <CitationChip key={i} citation={c} />
            ))}
          </div>
        )}

        <div style={{
          padding: "0.5rem 1.375rem",
          borderTop: "0.5px solid rgba(255,255,255,0.04)",
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          flexWrap: "wrap",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <SignalBars level={confLevel} color={confColor} />
            <span style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.68rem",
              color: confColor,
            }}>
              {isVerified ? "Verified" : "Partial match"}
            </span>
          </div>

          {intent && (
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--text-muted)" }}>
              {intent === "synthesis" ? "◈ Deep analysis" : "◉ Quick lookup"}
            </span>
          )}

          {citations && citations.length > 0 && (
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--text-muted)" }}>
              {citations.length === 1 ? "1 document" : `${citations.length} documents`}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function CitationChip({ citation }: { citation: Citation }) {
  const [showPreview, setShowPreview] = useState(false);
  return (
    <div
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setShowPreview(true)}
      onMouseLeave={() => setShowPreview(false)}
    >
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 5,
        background: "rgba(0,201,167,0.05)",
        border: "0.5px solid rgba(0,201,167,0.18)",
        borderRadius: 6,
        padding: "3px 10px",
        fontFamily: "var(--font-mono)",
        fontSize: "0.7rem",
        color: "rgba(0,201,167,0.7)",
        cursor: "pointer",
        transition: "all 0.15s",
      }}>
        <svg viewBox="0 0 24 24" width={10} height={10} fill="none" stroke="currentColor" strokeWidth="2" style={{ opacity: 0.5, flexShrink: 0 }}>
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14,2 14,8 20,8"/>
        </svg>
        <span>{citation.source}</span>
        {citation.section && (
          <span style={{ opacity: 0.45, fontSize: "0.65rem" }}>— {citation.section}</span>
        )}
      </div>
      {showPreview && citation.excerpt && (
        <div style={{
          position: "absolute",
          bottom: "calc(100% + 6px)",
          left: 0,
          zIndex: 50,
          background: "var(--surface-raised)",
          border: "0.5px solid rgba(0,201,167,0.2)",
          borderRadius: 10,
          padding: "0.75rem 1rem",
          width: 280,
          fontFamily: "var(--font-answer)",
          fontSize: "0.8rem",
          lineHeight: 1.6,
          color: "var(--text-secondary)",
          boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
          pointerEvents: "none",
        }}>
          <div style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.65rem",
            color: "var(--teal)",
            opacity: 0.7,
            marginBottom: "0.4rem",
          }}>
            {citation.source}
          </div>
          &ldquo;{citation.excerpt}...&rdquo;
        </div>
      )}
    </div>
  );
}

function SignalBars({ level, color }: { level: number; color: string }) {
  return (
    <div style={{ display: "flex", gap: 2, alignItems: "flex-end" }}>
      {[1, 2, 3].map(i => (
        <div key={i} style={{
          width: 3,
          height: i === 1 ? 5 : i === 2 ? 8 : 11,
          borderRadius: 1,
          background: i <= level ? color : "rgba(255,255,255,0.1)",
          transition: "background 0.3s",
        }} />
      ))}
    </div>
  );
}

function MarkdownBody({ text }: { text: string }) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;

  const renderInline = (s: string) => {
    const parts = s.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((p, j) =>
      p.startsWith("**") && p.endsWith("**")
        ? <strong key={j} style={{ fontWeight: 500, color: "var(--text-primary)" }}>{p.slice(2, -2)}</strong>
        : p
    );
  };

  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("- ") || line.startsWith("• ")) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && (lines[i].startsWith("- ") || lines[i].startsWith("• "))) {
        items.push(<li key={i} style={{ marginBottom: "0.2rem" }}>{renderInline(lines[i].slice(2))}</li>);
        i++;
      }
      elements.push(
        <ul key={`ul-${i}`} style={{ margin: "0.4rem 0 0.6rem 1.25rem", padding: 0 }}>
          {items}
        </ul>
      );
      continue;
    } else if (line.trim() === "") {
      i++;
      continue;
    } else {
      elements.push(
        <p key={i} style={{ marginBottom: "0.6rem" }}>{renderInline(line)}</p>
      );
    }
    i++;
  }

  return <>{elements}</>;
}

function OnboardingModal({ role, tenure, setRole, setTenure, onSave }: {
  role: string;
  tenure: string;
  setRole: (r: string) => void;
  setTenure: (t: string) => void;
  onSave: () => void;
}) {
  return (
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
        animation: "fadeUp 0.3s ease",
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
          Welcome to Savant
        </p>
        <h2 style={{
          fontWeight: 500,
          fontSize: "1.375rem",
          lineHeight: 1.3,
          marginBottom: "1.5rem",
          color: "var(--text-primary)",
        }}>
          Tell us about yourself
        </h2>

        {[
          { label: "Your role", value: role, setter: setRole, options: ["Manager", "Individual Contributor", "New Hire", "Other"] },
          { label: "Time at company", value: tenure, setter: setTenure, options: ["New Hire (0–30 days)", "1–6 Months", "6+ Months", "2+ Years"] },
        ].map(field => (
          <div key={field.label} style={{ marginBottom: "1.125rem" }}>
            <label style={{
              display: "block",
              fontSize: "0.8rem",
              color: "var(--text-secondary)",
              marginBottom: "0.4rem",
              fontWeight: 400,
            }}>
              {field.label}
            </label>
            <select
              value={field.value}
              onChange={e => field.setter(e.target.value)}
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
                cursor: "pointer",
                appearance: "none",
              }}
            >
              {field.options.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
        ))}

        <button
          onClick={onSave}
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
            marginTop: "0.5rem",
            transition: "opacity 0.15s",
            letterSpacing: "-0.01em",
          }}
          onMouseOver={e => (e.currentTarget as HTMLElement).style.opacity = "0.85"}
          onMouseOut={e => (e.currentTarget as HTMLElement).style.opacity = "1"}
        >
          Continue →
        </button>
      </div>

      <style>{`@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }`}</style>
    </div>
  );
}
