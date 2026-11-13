import { useEffect, useState } from "react";
import type { SummaryResponse } from "../api/types";

export default function HeatsView() {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [requests, setRequests] = useState<any[]>([]);
  const [scenario, setScenario] = useState("");
  useEffect(() => {
    fetch("/api/results/summary").then((r) => r.json()).then(setSummary).catch(() => {});
  }, []);
  useEffect(() => {
    const url = scenario ? `/api/results/requests?scenario=${scenario}` : "/api/results/requests";
    fetch(url).then((r) => r.json()).then(setRequests).catch(() => {});
  }, [scenario]);

  return (
    <div>
      <h1>Heats</h1>
      <div style={{ marginBottom: "1rem" }}>
        <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
          <option value="">all scenarios</option>
          <option value="steady">steady</option>
          <option value="burst">burst</option>
          <option value="sustained_overload">sustained_overload</option>
          <option value="rpm_halved">rpm_halved</option>
          <option value="daily_exhausted">daily_exhausted</option>
        </select>
      </div>
      {summary && (
        <div>
          <h2>Per scenario</h2>
          <table style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr><th>scenario</th><th>n</th><th>success%</th><th>rate-limited%</th><th>timed-out%</th><th>p50 FT</th><th>p95 FT</th><th>p99 FT</th><th>p50 total</th><th>retries/req</th><th>tokens/answer</th></tr>
            </thead>
            <tbody>
              {Object.entries(summary.scenarios).map(([name, b]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{b.n}</td>
                  <td>{b.success_pct?.toFixed(1)}</td>
                  <td>{b.rl_pct?.toFixed(1)}</td>
                  <td>{b.to_pct?.toFixed(1)}</td>
                  <td>{b.ft_p50}</td>
                  <td>{b.ft_p95}</td>
                  <td>{b.ft_p99}</td>
                  <td>{b.tot_p50}</td>
                  <td>{b.retries_per_req?.toFixed(2)}</td>
                  <td>{b.tokens_per_answer?.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h2>Per retrieval setting</h2>
          <table style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr><th>setting</th><th>n</th><th>success%</th><th>p50 FT</th><th>p95 FT</th><th>tokens/answer</th></tr>
            </thead>
            <tbody>
              {Object.entries(summary.settings).map(([name, b]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{b.n}</td>
                  <td>{b.success_pct?.toFixed(1)}</td>
                  <td>{b.ft_p50}</td>
                  <td>{b.ft_p95}</td>
                  <td>{b.tokens_per_answer?.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <h2>Requests ({requests.length})</h2>
      <div style={{ maxHeight: 300, overflow: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: "0.8rem" }}>
          <thead><tr><th>id</th><th>status</th><th>total_ms</th><th>grade</th></tr></thead>
          <tbody>
            {requests.slice(0, 50).map((r, i) => (
              <tr key={i}><td>{r.request_id}</td><td>{r.status}</td><td>{r.total_ms}</td><td>{r.answer_grade}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ opacity: 0.7, fontSize: "0.8rem" }}>Percentiles: numpy np.percentile (linear). Missing values excluded, not silently dropped.</p>
    </div>
  );
}
