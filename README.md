# The Relay

Multi-step LLM pipeline answering questions from 3 arXiv papers.

## Structure

- `relay/` - pipeline, scheduler, retrieval, API
- `standin/` - fake model server (Groq path shape)
- `loadgen/` - load testing
- `analysis/` - analysis scripts
- `data/` - data (raw ignored)
- `results/` - run results
- `eval/` - evaluation
- `scripts/` - list_models, check_tokens, repro_oversize
- `tests/` - tests

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
Copy-Item .env.example .env
# fill in GROQ_API_KEY, set MODEL_ID (see below)
```

## Settings (.env via python-dotenv)

| var | default | meaning |
|---|---|---|
| `GROQ_API_KEY` | - | real service key |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | real base URL (`/openai` segment required) |
| `STANDIN_BASE_URL` | `http://127.0.0.1:8001/openai/v1` | stand-in base URL (same path shape) |
| `MODEL_ID` | `standin-model` | model id; verified against `GET /models` for `source=real` |
| `REAL_DAILY_CALL_CAP` | `20` | safety cap, `source=real` |
| `REAL_DAILY_TOKEN_CAP` | `0` | safety cap, 0 = not set (reported as "not measured") |

No PowerShell env vars are needed; `dev.ps1` just starts the processes.

## Run

- Stand-in server: `uvicorn standin.server:app --port 8001` (config: `standin/config.json`)
  - Serves `POST /openai/v1/chat/completions`, `GET /openai/v1/models`, `GET /openai/v1/stats`, `GET /health`
  - Only base URL and key differ between stand-in and real service
- List models for your key: `python scripts/list_models.py` (prints id, owner, context window if the API returns it; no guessed limits — check https://console.groq.com/settings/limits for the chosen model)
- Token table: `python scripts/check_tokens.py --n 5 --setting chunk400-k5 --base-url http://127.0.0.1:8001/openai/v1`
- Calibrate stand-in: `python -m standin.calibrate` (reads `results/probe_raw.jsonl`)
- Probe Groq (dry-run first!): `python -m relay.probe --dry-run` then `python -m relay.probe --n 3 --max-requests 12`
- Ingest (download + extract + chunk): `python -m relay.ingest` (options: `--sizes 200 400 800`, `--force-download`)
  - Outputs: `data/raw/` (PDFs, gitignored), `data/extracted/*.json`, `data/chunks/chunks_*.jsonl`
- Pipeline: `python -m relay.pipeline --question "..."` (options: `--chunks`, `--k`, `--trace-out`, `--rpm`, `--tpm`, `--rpd`, `--give-up-time`, `--disable-retries`, `--small-model`)
- Loadgen: `python -m loadgen.run`
- Eval: `python -m eval.harness`
- Tests: `pytest`
- Oversized-estimate repro: `python scripts/repro_oversize.py --base-url http://127.0.0.1:8000`

### Model choice (record this — every limit in Table 1 belongs to this model)

- **No hardcoded Groq model.** `MODEL_ID` is read from `.env`; `scripts/list_models.py` lists what your key can use. `source=real` requests verify `MODEL_ID` against `GET /models` and fail fast with a readable error if absent (also fails fast if a call returns `model_decommissioned`/`model_not_found`; classified `model_endpoint_unreachable`, never retried).
- **Chosen model id: `openai/gpt-oss-120b`, set 2026-11-13.** Picked from `python scripts/list_models.py` output for this key (11 models; the chat-capable candidates were `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `allam-2-7b`). `openai/gpt-oss-120b` has a 131072-token context window (from the models response) and is listed in Groq's Structured Outputs docs as supporting strict mode; the pipeline sends `response_format: {"type": "json_object"}` (JSON mode), which Groq documents as working with streaming. Record this id and date in `results.csv` metadata when probing.
- **Groq limits are not guessed.** `GET /models` does not return RPM/TPM/RPD; read them from https://console.groq.com/settings/limits for the chosen model. `scripts/list_models.py` prints a reminder instead of inventing numbers.
- Groq deprecations: llama-3.1-8b-instant and llama-3.3-70b-versatile were announced for deprecation 2026-06-17 — and neither appears in this key's model list anymore (confirmed by `list_models.py` 2026-11-13).
- **Reasoning-model assumptions (record before probing):** `openai/gpt-oss-120b` is a reasoning model. Assumptions to verify on first real probe: (1) reasoning effort is not set by the pipeline (defaults to the provider's default), (2) if reasoning text is streamed, `first_token_ms` is the first streamed token of any kind (reasoning or answer) — the pipeline records whatever chunk arrives first, (3) JSON mode (`json_object`) constrains the final answer to JSON; the decide step depends on this. If a probe contradicts these, update this section with the observation.

### Token estimation

chars/4 (honest, simple). Estimator error is logged per call (estimated vs actual tokens_in/out) and corrected against the bucket: each call reserves `est_in + max_tokens`, and the bucket is corrected with actual usage afterward; the trace records the reservation and the correction. Validate against `usage.prompt_tokens` from real Groq responses; the stand-in computes usage with the same chars/4 rule, so stand-in-only error is not meaningful.

## Stand-in differences from the real service (deliberate)

The stand-in is a stand-in, not a clone of Groq's model behavior. Deliberate differences:

- **Step-aware canned replies.** The stand-in reads the `X-Relay-Step` request header (or detects the step from the system prompt) and returns output in the shape each step expects: valid JSON for `understand`/`decide`/`verify`, and a plausible answer citing `[paper_id, page, chunk_id]` triples parsed from the actual passages in the prompt for `draft`. The real model generates its own text.
- **Configurable failure injection.** `failures.malformed_json_rate` returns unparseable output (exercises the stricter-instruction retry and the `unparseable_model_output` error path); `failures.bad_citation_rate` cites a chunk id not in the passages (exercises the verify regenerate and refuse paths). The real service fails only naturally.
- **Configurable rate windows.** `limits.window_seconds` (default 60) is the sliding window counting input + output tokens, so you can match it to what your probe observes on the real service. The real window is not configurable.
- **Token accounting.** The stand-in computes `prompt_tokens` as `len(prompt)//4` — the same rule as the relay estimator — so estimator error measured against the stand-in is structurally near zero. Only real Groq `usage.prompt_tokens` says anything about estimator error.
- Latency is a sampled lognormal with load slowdown, not real model latency.

## Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173, proxies /api to :8000
npm run build     # outputs frontend/dist
npm run typecheck # tsc --noEmit
npm run test      # vitest run
```

Production: `npm run build` then serve via FastAPI. `relay/api.py` registers all `/api` routes first; unknown `/api/*` paths return JSON 404; unknown other paths serve `frontend/dist` (file if it exists, else the SPA index) for GET only. Run `.\prod.ps1`.

Dev (Windows): `.\dev.ps1` starts stand-in (:8001), relay API (:8000), and Vite (:5173).

## API error semantics

- `model_endpoint_unreachable` - model service down or does not serve `/chat/completions` (404/405/connection refused). Status `error`, never retried, UI shows a plain setup-problem message with raw detail behind a details toggle.
- `unparseable_model_output` - decide/understand output unparseable after one stricter retry. Status `error`, not a refusal.
- `cannot_fit_budget` - even one passage exceeds the token cap after degradation. Status `cannot_fit_budget`, never `timeout`.
- `timeout` - a genuine wait past the deadline (which gate blocked is named in the trace).
