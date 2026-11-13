import type { SSEEvent } from "./types";

export interface SSECallbacks {
  onEvent: (ev: SSEEvent) => void;
  onError?: (err: Error) => void;
}

export async function streamAsk(
  body: { question: string; source: "stand-in" | "real"; retrieval_setting: string },
  callbacks: SSECallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const lines = part.split("\n");
        let event = "message";
        let data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (data) {
          try {
            const parsed = JSON.parse(data);
            callbacks.onEvent({ type: event as SSEEvent["type"], data: parsed } as SSEEvent);
          } catch (e) {
            callbacks.onError?.(new Error("parse error: " + String(e)));
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
