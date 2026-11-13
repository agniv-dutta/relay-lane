import { describe, it, expect } from "vitest";
import { reduceEvent } from "../src/api/reducer";
import type { AskState } from "../src/api/useAsk";
import fixture from "./fixture.json";

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
  running: true,
  error: null,
};

describe("SSE event reducer", () => {
  it("reduces a recorded fixture", () => {
    let state = initial;
    for (const ev of fixture as any[]) {
      state = reduceEvent(state, ev as any, 0);
    }
    expect(state.streamedAnswer).toBe("Hello world");
    expect(state.final).not.toBeNull();
    expect(state.final?.status).toBe("ok");
    expect(state.steps.length).toBe(1);
    expect(state.steps[0].tokens_in).toBe(10);
    expect(state.firstTokenMs).not.toBeNull();
  });
});
