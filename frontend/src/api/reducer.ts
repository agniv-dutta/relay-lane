import type { SSEEvent } from "./types";
import type { AskState } from "./useAsk";

export function reduceEvent(prev: AskState, ev: SSEEvent, _t0: number): AskState {
  const next = { ...prev };
  switch (ev.type) {
    case "step_start": {
      const d = ev.data as { step: any; lane: number };
      next.currentLane = d.lane;
      next.steps = [...prev.steps, { step: d.step, lane: d.lane, start: 0, model_ms: null, tokens_in: null, tokens_out: null }];
      break;
    }
    case "step_end": {
      const d = ev.data as { step: any; lane: number; model_ms: number | null; tokens_in: number | null; tokens_out: number | null };
      next.steps = prev.steps.map((s) =>
        s.step === d.step && s.end === undefined ? { ...s, end: 0, model_ms: d.model_ms, tokens_in: d.tokens_in, tokens_out: d.tokens_out } : s,
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
      if (prev.firstTokenMs === null) next.firstTokenMs = 0;
      break;
    }
    case "token": {
      const d = ev.data as { text: string };
      next.streamedAnswer += d.text;
      break;
    }
    case "final": {
      const d = ev.data as any;
      next.final = d;
      next.totalMs = d.total_ms ?? 0;
      next.running = false;
      break;
    }
  }
  return next;
}
