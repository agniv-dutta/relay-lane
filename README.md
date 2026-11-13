# The Relay

Multi-step LLM pipeline answering questions from 3 arXiv papers.

## Structure

- `relay/` - pipeline, scheduler, retrieval
- `standin/` - fake model server
- `loadgen/` - load testing
- `analysis/` - analysis scripts
- `data/` - data (raw ignored)
- `results/` - run results
- `eval/` - evaluation
- `tests/` - tests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Copy-Item .env.example .env
# fill in GROQ_API_KEY
```

## Run

- Stand-in server: `uvicorn standin.server:app --reload` (config: `standin/config.json`)
- Calibrate stand-in: `python -m standin.calibrate` (reads `results/probe_raw.jsonl`)
- Probe Groq (dry-run first!): `python -m relay.probe --dry-run` then `python -m relay.probe --n 3 --max-requests 12`
- Ingest (download + extract + chunk): `python -m relay.ingest` (options: `--sizes 200 400 800`, `--force-download`)
  - Outputs: `data/raw/` (PDFs, gitignored), `data/extracted/*.json`, `data/chunks/chunks_*.jsonl`
- Pipeline: `python -m relay.pipeline --question "..."` (options: `--chunks`, `--k`, `--trace-out`, `--rpm`, `--tpm`, `--rpd`, `--give-up-time`, `--disable-retries`, `--small-model`)
- Loadgen: `python -m loadgen.run` (TODO)
- Token estimation uses chars/4 (honest, simple). Estimator error is logged per call (estimated vs actual tokens_in/out) and corrected against the bucket. Validate against `usage.prompt_tokens` from real Groq responses; stand-in uses the same chars/4 rule, so stand-in-only error is not meaningful.
- Eval: `python -m eval.run` (TODO)
- Tests: `pytest`

## Frontend

\\\ash
cd frontend
npm install
npm run dev       # http://localhost:5173, proxies /api to :8000
npm run build     # outputs frontend/dist
npm run typecheck # tsc --noEmit
\\\`n
Production: \
pm run build\ then serve via FastAPI (\elay.api\ mounts \rontend/dist\ at \/\). Run \.\prod.ps1\.

Dev (Windows): \.\dev.ps1\ starts stand-in (:8001), relay API (:8000), and Vite (:5173).
