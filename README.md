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
cp .env.example .env
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
- Analysis: `python -m analysis.make_tables` (reads `results/results.csv`)
- Eval: `python -m eval.run` (TODO)
- Tests: `pytest`
