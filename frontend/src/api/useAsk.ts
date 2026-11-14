import { useCallback, useRef, useState } from "react";
import { streamAsk } from "./sse";
import type { SSEEvent, FinalEvent, StepName } from "./types";

export interface StepTiming {
  step: StepName;
  lane: number;
  start: number;
  end?: number;
  model_ms: number | null;
  tokens_in: number | null;
  tokens_out: number | null;
}

export interface AskState {
  currentLane: number | null;
  steps: StepTiming[];
  limiterWaits: { ms: number; reason: string }[];
  retries: { attempt: number; backoff_ms: number; reason: string }[];
  degrades: { rung: number; reason: string }[];
  streamedAnswer: string;
  firstTokenMs: number | null;
  totalMs: number | null;
  final: FinalEvent | null;
  running: boolean;
  error: string | null;
}

const initial: AskState = {
  currentLane: null,
  steps: [],
  limiterWaits: [],
  retries: [],
  degrades: [],
  streamedAnswer: "",
  firstTokenMs: null,
  totalMs: null,
  final: null,
  running: false,
  error: null,
};

export function useAsk() {
  const [state, setState] = useState<AskState>(initial);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => setState(initial), []);

  const ask = useCallback(async (question: string, source: "stand-in" | "real", setting: string) => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setState({ ...initial, running: true });
    const t0 = performance.now();
    try {
      await streamAsk(
        { question, source, retrieval_setting: setting },
        {
          onEvent: (ev: SSEEvent) => {
            setState((prev) => {
              const next = { ...prev };
              switch (ev.type) {
                case "accepted":
                  break;
                case "step_start": {
                  const d = ev.data as { step: StepName; lane: number };
                  next.currentLane = d.lane;
                  next.steps = [...prev.steps, { step: d.step, lane: d.lane, start: performance.now() - t0, model_ms: null, tokens_in: null, tokens_out: null }];
                  break;
                }
                case "step_end": {
                  const d = ev.data as { step: StepName; lane: number; model_ms: number | null; tokens_in: number | null; tokens_out: number | null };
                  next.steps = prev.steps.map((s) =>
                    s.step === d.step && s.end === undefined ? { ...s, end: performance.now() - t0, model_ms: d.model_ms, tokens_in: d.tokens_in, tokens_out: d.tokens_out } : s,
                  );
                  break;
                }
                case "limiter_wait": {
                  const d = ev.data as { ms: number; reason: string };
                  next.limiterWaits = [...prev.limiterWaits, d];
                  break;
                }
                case "retry": {
                  const d = ev.data as { attempt: number; backoff_ms: number; reason: string };
                  next.retries = [...prev.retries, d];
                  break;
                }
                case "degrade": {
                  const d = ev.data as { rung: number; reason: string };
                  next.degrades = [...prev.degrades, d];
                  break;
                }
                case "first_token": {
                  if (prev.firstTokenMs === null) next.firstTokenMs = performance.now() - t0;
                  break;
                }
                case "token": {
                  const d = ev.data as { text: string };
                  next.streamedAnswer += d.text;
                  break;
                }
                case "final": {
                  const d = ev.data as FinalEvent;
                  next.final = d;
                  next.totalMs = d.total_ms ?? performance.now() - t0;
                  next.running = false;
                  break;
                }
              }
              return next;
            });
          },
          onError: (err) => setState((prev) => ({ ...prev, error: err.message, running: false })),
        },
        ac.signal,
      );
    } catch (e) {
      setState((prev) => ({ ...prev, error: String(e), running: false }));
    }
  }, []);

  const replay = useCallback((trace: any, speed: number) => {
    // simple replay: emit events with timeouts
    abortRef.current?.abort();
    setState({ ...initial, running: true });
    const t0 = performance.now();
    const events: any[] = [];
    // build from trace steps
    let acc = 0;
    for (const st of trace.steps || []) {
      events.push({ delay: acc, ev: { type: "step_start", data: { step: st.name, lane: 0 } } });
      acc += st.duration_ms / speed;
      events.push({ delay: acc, ev: { type: "step_end", data: { step: st.name, lane: 0, model_ms: st.model_time_ms, tokens_in: st.prompt_tokens, tokens_out: st.completion_tokens } } });
    }
    // final
    events.push({ delay: acc + 100, ev: { type: "final", data: { status: trace.refusal_reason === "cannot_fit_budget" ? "cannot_fit_budget" : trace.refused ? "refused" : "ok", answer: trace.final_answer, citations: [], refusal_reason: trace.refusal_reason, grade_hint: "", total_ms: trace.total_ms } } });

    (async () => {
      for (const e of events) {
        await new Promise((r) => setTimeout(r, Math.max(0, e.delay - (performance.now() - t0))));
        // not perfectly timed but ok
        setState((prev) => {
          const next = { ...prev };
          if (e.ev.type === "step_start") {
            const d = e.ev.data as { step: StepName; lane: number };
            next.currentLane = d.lane;
          } else if (e.ev.type === "final") {
            next.final = e.ev.data as FinalEvent;
            next.totalMs = (e.ev.data as FinalEvent).total_ms;
            next.running = false;
          }
          return next;
        });
      }
    })();
  }, []);

  return { state, ask, reset, replay };
}
