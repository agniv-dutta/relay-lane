import { useEffect, useRef, useState } from "react";
import type { LimitsResponse } from "../api/types";

function scopeWords(s: string | undefined): string {
  if (s === "real_service") return "real service";
  if (s === "stand_in") return "stand-in";
  if (s === "safety_cap") return "my safety cap";
  if (s === "not_measured") return "not measured";
  return s || "unknown";
}

export default function LimitsView() {
  const [limits, setLimits] = useState<LimitsResponse | null>(null);
  const [lastGood, setLastGood] = useState<LimitsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<any>(null);
  const [configText, setConfigText] = useState("");
  const [configError, setConfigError] = useState<string | null>(null);
  const [dailyExhausted, setDailyExhausted] = useState(false);
  const backoffRef = useRef(2000);
  const timerRef = useRef<number | null>(null);
  const hiddenRef = useRef(false);

  useEffect(() => {
    const poll = async () => {
      if (hiddenRef.current) return;
      try {
        const r = await fetch("/api/limits");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        setLimits(d);
        setLastGood(d);
        setError(null);
        backoffRef.current = 2000;
      } catch (e) {
        setError("can't reach the API");
        backoffRef.current = Math.min(backoffRef.current * 2, 30000);
      }
      timerRef.current = window.setTimeout(poll, backoffRef.current);
    };
    const onVis = () => {
      hiddenRef.current = document.hidden;
      if (!document.hidden) {
        backoffRef.current = 2000;
        if (timerRef.current) clearTimeout(timerRef.current);
        poll();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    poll();
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  useEffect(() => {
    fetch("/api/standin/config").then((r) => r.json()).then((d) => { setConfig(d); setConfigText(JSON.stringify(d, null, 2)); }).catch(() => {});
  }, []);

  const display = limits || lastGood;
  const dimmed = !limits && !!lastGood;

  const validateJson = (text: string): any | null => {
    try {
      const parsed = JSON.parse(text);
      setConfigError(null);
      return parsed;
    } catch (e) {
      setConfigError((e as Error).message);
      return null;
    }
  };

  const parsedConfig = (() => {
    try { return JSON.parse(configText); } catch { return null; }
  })();

  const changed = JSON.stringify(parsedConfig) !== JSON.stringify(config);

  const diff = (() => {
    if (!parsedConfig || !config) return "";
    const lines: string[] = [];
    const allKeys = new Set([...Object.keys(parsedConfig), ...Object.keys(config)]);
    for (const k of allKeys) {
      const a = JSON.stringify(config[k]);
      const b = JSON.stringify(parsedConfig[k]);
      if (a !== b) lines.push(`${k}: ${a} -> ${b}`);
    }
    return lines.join("\n");
  })();

  const saveConfig = async () => {
    const parsed = validateJson(configText);
    if (!parsed) return;
    await fetch("/api/standin/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(parsed) });
    setConfig(parsed);
    setConfigText(JSON.stringify(parsed, null, 2));
  };

  const resetConfig = async () => {
    const r = await fetch("/api/standin/config");
    const d = await r.json();
    setConfig(d);
    setConfigText(JSON.stringify(d, null, 2));
    setConfigError(null);
  };

  const bar = (lim: any, label: string) => {
    if (!lim || lim.limit === null || lim.limit === undefined) {
      return (
        <div style={{ marginBottom: "1rem", opacity: dimmed ? 0.5 : 1 }}>
          <div>{label}: not measured {lim?.error ? `(${lim.error})` : ""}</div>
        </div>
      );
    }
    const used = lim.used ?? 0;
    const limit = lim.limit;
    const pct = limit > 0 ? (used / limit) * 100 : 0;
    const low = pct > 85;
    return (
      <div style={{ marginBottom: "1rem", opacity: dimmed ? 0.5 : 1 }}>
        <div>{label} [{scopeWords(lim.scope)}]: {used} / {limit} (remaining {lim.remaining}){lim.resets_at ? ` resets ${new Date(lim.resets_at).toLocaleString()}` : ""}</div>
        <div style={{ height: 12, background: "var(--color-infield)", width: "100%" }}>
          <div style={{ height: 12, width: `${Math.min(100, pct)}%`, background: low ? "var(--color-false)" : "var(--color-baton)" }} />
        </div>
      </div>
    );
  };

  return (
    <div>
      <h1>Limits</h1>
      {error && <div style={{ color: "var(--color-false)", marginBottom: "1rem" }}>{error} (showing last good values)</div>}
      {display && (
        <div>
          {bar(display.rpm, "Per-minute requests (RPM)")}
          {bar(display.tpm, "Per-minute tokens (TPM)")}
          {bar(display.real_call_cap, "Real-request cap (daily calls)")}
          {bar(display.real_token_cap, "Real-token cap (daily tokens)")}
          <div style={{ opacity: dimmed ? 0.5 : 1, fontSize: "0.9rem" }}>
            In-flight: {display.in_flight ?? "not tracked"} · Queue length: {display.queue_length ?? "not tracked"}
          </div>
        </div>
      )}
      <h2>Degradation ladder</h2>
      <ol>
        <li>full retrieval</li>
        <li>fewer passages</li>
        <li>skip the model verify pass (keep the code check)</li>
        <li>smaller model</li>
        <li>honest "try later" reply</li>
      </ol>
      {display?.real_request_cap?.limit && display.real_request_cap.scope === "stand_in" && (
        <div style={{ color: "var(--color-false)", marginTop: "1rem" }}>Warning: stand-in is active, not the real service.</div>
      )}
      <h2>Stand-in config</h2>
      <div style={{ position: "relative", paddingBottom: "4rem" }}>
        <textarea
          style={{ width: "100%", height: 200 }}
          value={configText}
          onChange={(e) => { setConfigText(e.target.value); validateJson(e.target.value); }}
        />
        {configError && <div style={{ color: "var(--color-false)" }}>Invalid JSON: {configError}</div>}
        {diff && <pre style={{ background: "var(--color-infield)", padding: "0.5rem", fontSize: "0.8rem" }}>{diff}</pre>}
        <div style={{ position: "sticky", bottom: 0, background: "var(--color-mondo)", padding: "0.5rem", display: "flex", gap: "0.5rem" }}>
          <button onClick={saveConfig} disabled={!changed || !!configError}>Save</button>
          <button onClick={resetConfig}>Reset</button>
        </div>
      </div>
      <div style={{ marginTop: "1rem" }}>
        <label>
          <input type="checkbox" checked={dailyExhausted} onChange={(e) => setDailyExhausted(e.target.checked)} /> Trigger daily-exhausted state (dev)
        </label>
      </div>
    </div>
  );
}
