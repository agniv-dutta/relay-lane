import { useEffect, useState } from "react";

const PAPERS = [
  { id: "1706.03762", version: "v7" },
  { id: "2410.00037", version: "v1" },
  { id: "2507.07955", version: "v2" },
];

export default function PapersView() {
  const [size, setSize] = useState(400);
  const [query, setQuery] = useState("");
  const [chunks, setChunks] = useState<any[]>([]);

  const search = async () => {
    const r = await fetch(`/api/chunks/search?q=${encodeURIComponent(query)}&size=${size}`);
    const data = await r.json();
    setChunks(data);
  };

  return (
    <div>
      <h1>Papers</h1>
      <div style={{ marginBottom: "1rem" }}>
        <label>Chunking setting: </label>
        <select value={size} onChange={(e) => setSize(Number(e.target.value))}>
          <option value={200}>200</option>
          <option value={400}>400</option>
          <option value={800}>800</option>
        </select>
      </div>
      <div style={{ marginBottom: "1rem" }}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="search chunks" />
        <button onClick={search} style={{ marginLeft: "0.5rem" }}>Search</button>
      </div>
      <div>
        {PAPERS.map((p) => (
          <div key={p.id} style={{ marginBottom: "0.5rem" }}>
            <strong>{p.id}</strong> {p.version}
          </div>
        ))}
      </div>
      <div style={{ marginTop: "1rem" }}>
        {chunks.map((c) => (
          <div key={c.chunk_id} style={{ marginBottom: "0.5rem", padding: "0.5rem", background: "var(--color-infield)" }}>
            <div style={{ fontSize: "0.8rem", opacity: 0.8 }}>{c.chunk_id} (score {c.score})</div>
            <div>{c.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
