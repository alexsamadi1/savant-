"use client";

import { useState, useRef, useEffect, useCallback, FormEvent } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  content: string;
}

/* ------------------------------------------------------------------ */
/*  SSE stream reader for POST /chat/stream                           */
/* ------------------------------------------------------------------ */

async function streamChat(
  tenant: string,
  messages: Message[],
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: string) => void,
) {
  try {
    const resp = await fetch(`${BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant, messages, model: "gpt-4o" }),
    });
    if (!resp.ok) {
      onError(`Server error: ${resp.status}`);
      return;
    }
    if (!resp.body) {
      onError("No stream body");
      return;
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        try {
          const event = JSON.parse(line.slice(5).trim());
          if (event.done) {
            onDone();
            return;
          }
          if (event.token) {
            onToken(event.token);
          }
        } catch {
          /* skip malformed lines */
        }
      }
    }
    onDone();
  } catch (e: unknown) {
    onError(e instanceof Error ? e.message : String(e));
  }
}

/* ------------------------------------------------------------------ */
/*  Page component                                                    */
/* ------------------------------------------------------------------ */

export default function ChatPage() {
  const params = useParams();
  const tenant = params.tenant as string;

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = useCallback(
    (e: FormEvent) => {
      e.preventDefault();
      const text = input.trim();
      if (!text || streaming) return;

      const userMsg: Message = { role: "user", content: text };
      const updated = [...messages, userMsg];
      setMessages(updated);
      setInput("");
      setStreaming(true);

      // Add an empty assistant message that we'll stream into
      const assistantIdx = updated.length;
      setMessages([...updated, { role: "assistant", content: "" }]);

      streamChat(
        tenant,
        updated,
        (token) => {
          setMessages((prev) => {
            const copy = [...prev];
            copy[assistantIdx] = {
              ...copy[assistantIdx],
              content: copy[assistantIdx].content + token,
            };
            return copy;
          });
        },
        () => setStreaming(false),
        (err) => {
          setMessages((prev) => {
            const copy = [...prev];
            copy[assistantIdx] = {
              ...copy[assistantIdx],
              content: `Error: ${err}`,
            };
            return copy;
          });
          setStreaming(false);
        },
      );
    },
    [input, messages, streaming, tenant],
  );

  /* ---- Styles ---- */

  const userBubble: React.CSSProperties = {
    alignSelf: "flex-end",
    background: "var(--teal)",
    color: "var(--bg)",
    borderRadius: "14px 14px 4px 14px",
    padding: "0.65rem 1rem",
    maxWidth: "75%",
    fontSize: "0.88rem",
    lineHeight: 1.6,
    fontFamily: "var(--font-ui)",
  };

  const assistantBubble: React.CSSProperties = {
    alignSelf: "flex-start",
    background: "var(--surface)",
    color: "var(--text-primary)",
    borderRadius: "14px 14px 14px 4px",
    borderLeft: "3px solid var(--teal)",
    padding: "0.65rem 1rem",
    maxWidth: "75%",
    fontSize: "0.88rem",
    lineHeight: 1.7,
    fontFamily: "var(--font-answer)",
    whiteSpace: "pre-wrap",
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        maxWidth: 760,
        margin: "0 auto",
        padding: "0 1rem",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          padding: "1rem 0",
          borderBottom: "1px solid var(--border)",
          flexShrink: 0,
        }}
      >
        <Link
          href={`/dashboard/${tenant}`}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            width: 32,
            height: 32,
            borderRadius: 8,
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
            textDecoration: "none",
            fontSize: "0.9rem",
            transition: "border-color 0.15s",
          }}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M19 12H5M12 19l-7-7 7-7" />
          </svg>
        </Link>
        <div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.72rem",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--teal)",
            }}
          >
            Follow-up Chat
          </div>
          <div
            style={{
              fontSize: "0.8rem",
              color: "var(--text-secondary)",
              marginTop: "0.1rem",
            }}
          >
            {tenant}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          padding: "1.25rem 0",
        }}
      >
        {messages.length === 0 && (
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: "0.75rem",
              color: "var(--text-muted)",
            }}
          >
            <div style={{ fontSize: "1.5rem", opacity: 0.4 }}>?</div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.78rem",
                letterSpacing: "0.04em",
              }}
            >
              Ask a question about your analysis
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={msg.role === "user" ? userBubble : assistantBubble}
          >
            {msg.content}
            {msg.role === "assistant" && streaming && i === messages.length - 1 && (
              <span
                style={{
                  display: "inline-block",
                  width: 6,
                  height: 14,
                  background: "var(--teal)",
                  marginLeft: 2,
                  animation: "blink 0.8s infinite",
                  verticalAlign: "text-bottom",
                }}
              />
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          gap: "0.5rem",
          padding: "0.75rem 0 1.25rem",
          borderTop: "1px solid var(--border)",
          flexShrink: 0,
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a follow-up question..."
          disabled={streaming}
          style={{
            flex: 1,
            padding: "0.7rem 1rem",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            color: "var(--text-primary)",
            fontFamily: "var(--font-ui)",
            fontSize: "0.88rem",
            outline: "none",
            transition: "border-color 0.15s",
          }}
          onFocus={(e) =>
            (e.target.style.borderColor = "var(--border-bright)")
          }
          onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
        />
        <button
          type="submit"
          disabled={streaming || !input.trim()}
          style={{
            padding: "0.7rem 1.25rem",
            background:
              streaming || !input.trim() ? "var(--surface)" : "var(--teal)",
            color:
              streaming || !input.trim()
                ? "var(--text-muted)"
                : "var(--bg)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            fontFamily: "var(--font-ui)",
            fontWeight: 600,
            fontSize: "0.85rem",
            cursor:
              streaming || !input.trim() ? "not-allowed" : "pointer",
            transition: "all 0.15s",
          }}
        >
          {streaming ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
