"use client";
import { useState, useCallback } from "react";
import { StreamEvent, Citation, UserProfile } from "./types";
import { getStreamUrl, buildStreamBody } from "./api";

interface StreamState {
  tokens: string;
  citations: Citation[];
  grounded: boolean | null;
  confidence: number | null;
  intent: string | null;
  status: "idle" | "searching" | "streaming" | "done" | "error";
  latency: number | null;
  followUps: string[];
}

export function useStream() {
  const [state, setState] = useState<StreamState>({
    tokens: "", citations: [], grounded: null, confidence: null,
    intent: null, status: "idle", latency: null, followUps: [],
  });

  const ask = useCallback(async (question: string, profile: UserProfile, model = "gpt-4o-mini") => {
    const start = Date.now();
    setState({ tokens: "", citations: [], grounded: null, confidence: null, intent: null, status: "searching", latency: null, followUps: [] });
    try {
      const response = await fetch(getStreamUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: buildStreamBody(question, profile, model),
      });
      if (!response.body) throw new Error("No stream body");
      setState(s => ({ ...s, status: "streaming" }));
      const reader = response.body.getReader();
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
            const event: StreamEvent = JSON.parse(line.slice(5).trim());
            if (event.token && !event.done) {
              setState(s => ({ ...s, tokens: s.tokens + event.token! }));
            }
            if (event.done) {
              setState(s => ({ ...s, status: "done", citations: event.citations || [], grounded: event.grounded ?? true, confidence: event.rerank_confidence ?? null, intent: event.intent || null, latency: (Date.now() - start) / 1000, followUps: event.follow_ups || [] }));
            }
          } catch {}
        }
      }
    } catch {
      setState(s => ({ ...s, status: "error" }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({ tokens: "", citations: [], grounded: null, confidence: null, intent: null, status: "idle", latency: null, followUps: [] });
  }, []);

  return { ...state, ask, reset };
}
