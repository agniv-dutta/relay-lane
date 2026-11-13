import { useState } from "react";
import { useAsk } from "../api/useAsk";

const LANES = [
  { n: 1, name: "understand" },
  { n: 2, name: "retrieve" },
  { n: 3, name: "decide" },
  { n: 4, name: "draft" },
  { n: 5, name: "verify" },
];

export default function TrackView() {
  const { state, ask, reset } = useAsk();
  const [question, setQuestion] = useState("");
  const [source, setSource] = useState<"stand-in" | "real">("stand-in");
  const [setting, setSetting] = useState("chunk400_k5_norerank");
  const [realUsed] = useState(0);
  const [realCap] = useState(20);
  const [hasRun, setHasRun] = useState(false);

  const run = () => {
    if (!question.trim()) return;
    setHasRun(true);
    ask(question, source, setting);
  };

  const laneH = 44;
  const topPad = 12;
  const leftPad = 150;
  const rightPad = 24;
  const totalH = topPad + LANES.length * laneH + 8;
  const barW = 600; // reference width for scaling

  // current lane index (1-based lane number)
  const currentLaneNum = state.currentLane ?? null;

  // first-token x position (scaled)
  const totalMs = state.totalMs ?? state.firstTokenMs ?? 1;
  const ftX = state.firstTokenMs !== null
    ? leftPad + Math.min((state.firstTokenMs / Math.max(totalMs, 1)) * (barW - leftPad - rightPad), barW - leftPad - rightPad)
    : null;

  // photo-finish segments
  const waitingMs = state.limiterWaits.reduce((a, w) => a + (w.ms || 0), 0);
  const retryingMs = state.retries.reduce((a, r) => a + (r.backoff_ms || 0), 0);
  const workingMs = Math.max(0, (state.totalMs || 0) - waitingMs - retryingMs);
  const stripTotal = Math.max(1, waitingMs + workingMs + retryingMs);
  const waitingW = (waitingMs / stripTotal) * barW;
  const workingW = (workingMs / stripTotal) * barW;
  const retryingW = (retryingMs / stripTotal) * barW;
  const ftStripX = state.firstTokenMs !== null ? (state.firstTokenMs / stripTotal) * barW : null;

  // countdown for latest limiter wait
  const latestWait = state.limiterWaits[state.limiterWaits.length - 1];

  return (
    <div>
      <h1>Track</h1>
      <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "1rem", flexWrap: "wrap" }}>
        <input
          style={{ flex: 1, minWidth: 200 }}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about the papers"
        />
        <select value={source} onChange={(e) => setSource(e.target.value as any)}>
          <option value="stand-in">stand-in</option>
          <option value="real">real</option>
        </select>
        <select value={setting} onChange={(e) => setSetting(e.target.value)}>
          <option value="chunk200_k5_norerank">chunk200 k5</option>
          <option value="chunk400_k5_norerank">chunk400 k5</option>
          <option value="chunk800_k5_norerank">chunk800 k5</option>
          <option value="chunk400_k10_norerank">chunk400 k10</option>
          <option value="chunk400_k5_rerank">chunk400 k5 rerank</option>
        </select>
        <button onClick={run} disabled={state.running}>Run the question</button>
        {source === "real" && <span>real requests used today: {realUsed} of {realCap}</span>}
      </div>

      {!hasRun && (
        <div style={{ marginBottom: "1rem", opacity: 0.8 }}>
          <p>Ask a question about the three papers to see the request move through the five lanes.</p>
          <button onClick={() => { setQuestion("What is scaled dot-product attention?"); }}>Try an example question</button>
        </div>
      )}

      <svg width="100%" height={totalH} style={{ background: "var(--color-mondo)", display: "block" }}>
        {/* lanes as bands */}
        {LANES.map((lane, i) => {
          const y = topPad + i * laneH;
          const tint = i % 2 === 0 ? "rgba(12,26,71,0.35)" : "rgba(12,26,71,0.15)";
          return (
            <g key={lane.n}>
              <rect x={leftPad} y={y} width={barW - leftPad - rightPad} height={laneH} fill={tint} />
              <line x1={leftPad} y1={y} x2={barW - rightPad} y2={y} stroke="var(--color-lane)" strokeWidth="1" />
              <line x1={leftPad} y1={y + laneH} x2={barW - rightPad} y2={y + laneH} stroke="var(--color-lane)" strokeWidth="1" />
              <text x={8} y={y + laneH / 2 + 5} fill="var(--color-lane)" fontSize="13" fontFamily="var(--font-display)" fontWeight="700">
                {lane.n} {lane.name}
              </text>
            </g>
          );
        })}
        {/* start line */}
        <line x1={leftPad} y1={topPad} x2={leftPad} y2={topPad + LANES.length * laneH} stroke="var(--color-baton)" strokeWidth="2" />
        {/* finish line */}
        <line x1={barW - rightPad} y1={topPad} x2={barW - rightPad} y2={topPad + LANES.length * laneH} stroke="var(--color-lane)" strokeWidth="2" strokeDasharray="4 3" />
        {/* idle baton at start line */}
        {!state.running && !state.final && (
          <rect x={leftPad + 2} y={topPad + 2} width={18} height={laneH - 4} fill="var(--color-baton)" opacity={0.6} rx={2} />
        )}
        {/* baton in current lane (pulsing bar, no invented progress) */}
        {state.running && currentLaneNum !== null && (
          <rect
            x={leftPad + 6}
            y={topPad + (currentLaneNum - 1) * laneH + 6}
            width={28}
            height={laneH - 12}
            fill="var(--color-baton)"
            opacity={0.9}
          >
            <animate attributeName="opacity" values="0.9;0.4;0.9" dur="1.2s" repeatCount="indefinite" />
          </rect>
        )}
        {/* hold line + countdown for limiter_wait */}
        {latestWait && state.running && (
          <g>
            <line x1={leftPad + 40} y1={topPad} x2={leftPad + 40} y2={topPad + LANES.length * laneH} stroke="var(--color-false)" strokeWidth="2" />
            <text x={leftPad + 44} y={topPad + 14} fill="var(--color-false)" fontSize="12" fontFamily="var(--font-display)">
              held: {latestWait.reason} { (latestWait.ms / 1000).toFixed(1) }s
            </text>
          </g>
        )}
        {/* first-token vertical line across all lanes */}
        {ftX !== null && (
          <line x1={ftX} y1={topPad} x2={ftX} y2={topPad + LANES.length * laneH} stroke="var(--color-baton)" strokeWidth="2" />
        )}
        {/* retry ticks */}
        {state.retries.map((r, i) => (
          <line key={i} x1={leftPad + 20 + i * 6} y1={topPad + (r.attempt ?? 1) * 0 + 2} x2={leftPad + 20 + i * 6} y2={topPad + 8} stroke="var(--color-false)" strokeWidth="2" />
        ))}
      </svg>

      {state.limiterWaits.length > 0 && (
        <div style={{ color: "var(--color-false)", marginTop: "0.5rem" }}>
          {state.limiterWaits.map((w, i) => (
            <div key={i}>held: {w.reason} { (w.ms / 1000).toFixed(1) }s</div>
          ))}
        </div>
      )}
      {state.retries.length > 0 && (
        <div style={{ color: "var(--color-false)" }}>
          {state.retries.map((r, i) => <div key={i}>retry {r.attempt}: {r.reason} {r.backoff_ms}ms</div>)}
        </div>
      )}
      {state.degrades.length > 0 && (
        <div style={{ color: "var(--color-baton)" }}>
          {state.degrades.map((d, i) => <div key={i}>degraded rung {d.rung}: {d.reason}</div>)}
        </div>
      )}

      <div style={{ marginTop: "1rem", background: "var(--color-infield)", padding: "1rem" }}>
        <div style={{ fontFamily: "var(--font-body)", maxWidth: "66ch" }}>
          {state.streamedAnswer}
        </div>
        {state.final && (
          <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", opacity: 0.8 }}>
            {state.final.refusal_reason && <div>{state.final.refusal_reason}</div>}
            <div>tokens in/out: {state.steps.reduce((a, s) => a + (s.tokens_in || 0), 0)} / {state.steps.reduce((a, s) => a + (s.tokens_out || 0), 0)}</div>
            <div>setting: {setting}</div>
            <div>first token: {state.firstTokenMs?.toFixed(0)} ms, total: {state.totalMs?.toFixed(0)} ms</div>
          </div>
        )}
        {state.error && <div style={{ color: "var(--color-false)" }}>{state.error}</div>}
      </div>

      {/* photo-finish strip */}
      {state.totalMs !== null && state.totalMs > 0 && (
        <div style={{ marginTop: "1rem" }}>
          <svg width={barW} height={24}>
            <rect x={0} y={0} width={waitingW} height={24} fill="var(--color-lane)" />
            <rect x={waitingW} y={0} width={workingW} height={24} fill="var(--color-baton)" />
            <rect x={waitingW + workingW} y={0} width={retryingW} height={24} fill="var(--color-false)" />
            {ftStripX !== null && (
              <line x1={ftStripX} y1={0} x2={ftStripX} y2={24} stroke="var(--color-baton)" strokeWidth="2" />
            )}
          </svg>
          <div style={{ fontSize: "0.8rem", opacity: 0.8 }}>
            first token at {state.firstTokenMs?.toFixed(0)} ms, {state.firstTokenMs !== null && state.totalMs !== null && state.firstTokenMs < (state.totalMs / 2) ? "before" : "after"} the wait
            {" · "}waiting {waitingMs.toFixed(0)}ms · working {workingMs.toFixed(0)}ms · retrying {retryingMs.toFixed(0)}ms
          </div>
        </div>
      )}

      {hasRun && (
        <div style={{ marginTop: "1rem" }}>
          <button onClick={reset}>Reset</button>
        </div>
      )}

      <div aria-live="polite" style={{ position: "absolute", left: -9999 }}>
        {state.steps.map((s) => s.step).join(", ")}
      </div>
    </div>
  );
}
