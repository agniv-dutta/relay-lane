"""FastAPI backend exposing the pipeline to the frontend."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from relay.model_client import ModelClient
from relay.pipeline import run_pipeline
from relay.retrieval import BM25Retriever, OverlapReranker, NoOpReranker
from relay.scheduler import Scheduler, DeadlineExceeded
from analysis.make_tables import load_rows, group_stats, compute_block


def parse_setting(s: str):
    import re
    chunk_size, k, rerank = 400, 5, False
    m = re.search(r"chunk(\d+)", s)
    if m:
        chunk_size = int(m.group(1))
    m = re.search(r"k(\d+)", s)
    if m:
        k = int(m.group(1))
    if "rerank" in (s or "").lower() and "norerank" not in (s or "").lower() and "no_rerank" not in (s or "").lower() and "no-rerank" not in (s or "").lower():
        rerank = True
    return chunk_size, k, rerank


app = FastAPI(title="relay-lane API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Real-source hard cap (in-memory, per process)
REAL_CALL_CAP = int(os.getenv("REAL_DAILY_CALL_CAP", "20"))
REAL_TOKEN_CAP = int(os.getenv("REAL_DAILY_TOKEN_CAP", "0"))

_real_used = {"calls": 0, "tokens": 0, "day": time.strftime("%Y-%m-%d")}


def _real_cap_check(est_tokens=0):
    global _real_used
    today = time.strftime("%Y-%m-%d")
    if today != _real_used["day"]:
        _real_used = {"calls": 0, "tokens": 0, "day": today}
    if _real_used["calls"] >= REAL_CALL_CAP:
        return False
    if REAL_TOKEN_CAP > 0 and _real_used["tokens"] + est_tokens > REAL_TOKEN_CAP:
        return False
    _real_used["calls"] += 1
    _real_used["tokens"] += est_tokens
    return True


# In-memory trace store
TRACES: dict[str, dict] = {}


_RETRIEVER_CACHE: dict[tuple, BM25Retriever] = {}


def _chunks_for(size: int) -> str:
    return f"data/chunks/chunks_{size}.jsonl"


def _get_retriever(chunks_path: str, rerank: bool) -> BM25Retriever:
    key = (chunks_path, rerank)
    if key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[key] = BM25Retriever(chunks_path, reranker=OverlapReranker() if rerank else NoOpReranker())
    return _RETRIEVER_CACHE[key]


def _find_chunk(chunk_id: str):
    for p in Path("data/chunks").glob("chunks_*.jsonl"):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("chunk_id") == chunk_id:
                    return obj
    return None


def _probe_limits():
    """Read RPM/TPM limits from results/probe_raw.jsonl if available."""
    p = Path("results/probe_raw.jsonl")
    if not p.exists():
        return None, None
    from collections import Counter
    rpm_vals, tpm_vals = [], []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                h = rec.get("rate_limit_headers") or {}
                for k, v in h.items():
                    kl = k.lower()
                    if kl == "x-ratelimit-limit-requests":
                        try:
                            rpm_vals.append(int(float(v)))
                        except (ValueError, TypeError):
                            pass
                    elif kl == "x-ratelimit-limit-tokens":
                        try:
                            tpm_vals.append(int(float(v)))
                        except (ValueError, TypeError):
                            pass
    except Exception:
        return None, None
    rpm = Counter(rpm_vals).most_common(1)[0][0] if rpm_vals else None
    tpm = Counter(tpm_vals).most_common(1)[0][0] if tpm_vals else None
    return rpm, tpm


def _standin_limits():
    try:
        cfg = json.loads(Path("standin/config.json").read_text(encoding="utf-8"))
        L = cfg.get("limits", {})
        return L.get("rpm"), L.get("tpm")
    except Exception:
        return None, None


@app.get("/api/health")
async def health():
    # stand-in reachability
    standin_reachable = False
    try:
        import httpx
        base = os.getenv("STANDIN_BASE_URL", "http://localhost:8000")
        async with httpx.AsyncClient(base_url=base, timeout=2.0) as c:
            r = await c.get("/health")
            standin_reachable = r.status_code == 200
    except Exception:
        standin_reachable = False
    return {
        "ok": True,
        "standin_reachable": standin_reachable,
        "results_csv_present": Path("results/results.csv").exists(),
        "real_requests_used_today": _real_used["calls"],
        "real_cap": REAL_CALL_CAP,
    }


@app.get("/api/limits")
async def limits():
    rpm_probe, tpm_probe = _probe_limits()
    rpm_stand, tpm_stand = _standin_limits()

    # RPM
    if rpm_probe is not None:
        rpm_limit = rpm_probe
        rpm_scope = "real_service"
        rpm_error = None
    elif rpm_stand is not None:
        rpm_limit = rpm_stand
        rpm_scope = "stand_in"
        rpm_error = None
    else:
        rpm_limit = None
        rpm_scope = "not_measured"
        rpm_error = "not measured (run probe)"

    if tpm_probe is not None:
        tpm_limit = tpm_probe
        tpm_scope = "real_service"
        tpm_error = None
    elif tpm_stand is not None:
        tpm_limit = tpm_stand
        tpm_scope = "stand_in"
        tpm_error = None
    else:
        tpm_limit = None
        tpm_scope = "not_measured"
        tpm_error = "not measured (run probe)"

    resets_at = None  # not tracked globally

    return {
        "rpm": {
            "scope": rpm_scope,
            "used": None,
            "remaining": None,
            "limit": rpm_limit,
            "resets_at": resets_at,
            "error": rpm_error if rpm_limit is None else "usage not tracked globally",
        },
        "tpm": {
            "scope": tpm_scope,
            "used": None,
            "remaining": None,
            "limit": tpm_limit,
            "resets_at": resets_at,
            "error": tpm_error if tpm_limit is None else "usage not tracked globally",
        },
        "real_call_cap": {
            "scope": "safety_cap",
            "used": _real_used["calls"],
            "remaining": REAL_CALL_CAP - _real_used["calls"],
            "limit": REAL_CALL_CAP,
            "resets_at": None,
            "error": None,
        },
        "real_token_cap": {
            "scope": "safety_cap",
            "used": _real_used["tokens"],
            "remaining": (REAL_TOKEN_CAP - _real_used["tokens"]) if REAL_TOKEN_CAP > 0 else None,
            "limit": REAL_TOKEN_CAP if REAL_TOKEN_CAP > 0 else None,
            "resets_at": None,
            "error": "not measured" if REAL_TOKEN_CAP == 0 else None,
        },
        "in_flight": None,
        "queue_length": None,
        "in_flight_error": "not tracked globally",
        "queue_length_error": "not tracked globally",
    }


@app.get("/api/results/summary")
async def results_summary():
    rows = load_rows(Path("results/results.csv")) if Path("results/results.csv").exists() else []
    # per scenario + per retrieval setting
    out = {"scenarios": {}, "settings": {}}
    for name, grp in group_stats(rows, "scenario"):
        b = compute_block(grp)
        out["scenarios"][name] = b
    for name, grp in group_stats(rows, "retrieval_setting"):
        b = compute_block(grp)
        out["settings"][name] = b
    return out


@app.get("/api/results/requests")
async def results_requests(scenario: str = None):
    path = Path("results/results.csv")
    if not path.exists():
        return []
    rows = load_rows(path)
    if scenario:
        rows = [r for r in rows if r.get("scenario") == scenario]
    return rows


@app.get("/api/trace/{request_id}")
async def get_trace(request_id: str):
    if request_id in TRACES:
        return TRACES[request_id]
    # try file
    for p in Path("results").glob(f"trace*{request_id}*.json"):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    # any trace file
    for p in Path("results").glob("trace_*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("question") and request_id in str(p):
                return data
        except Exception:
            pass
    return JSONResponse(status_code=404, content={"detail": "trace not found"})


@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    obj = _find_chunk(chunk_id)
    if obj is None:
        return JSONResponse(status_code=404, content={"detail": "chunk not found"})
    return obj


@app.get("/api/standin/config")
async def get_standin_config():
    p = Path(os.getenv("STANDIN_CONFIG", "standin/config.json"))
    if not p.exists():
        p = Path("standin/config.json")
    return json.loads(p.read_text(encoding="utf-8"))


@app.post("/api/standin/config")
async def set_standin_config(request: Request):
    body = await request.json()
    p = Path(os.getenv("STANDIN_CONFIG", "standin/config.json"))
    if not p.exists():
        p = Path("standin/config.json")
    p.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"ok": True}


@app.post("/api/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question", "")
    source = body.get("source", "stand-in")
    retrieval_setting = body.get("retrieval_setting", "chunk400_k5_norerank")

    chunk_size, k, rerank = parse_setting(retrieval_setting)

    if source == "real":
        if not _real_cap_check(1000):
            return JSONResponse(status_code=429, content={"detail": "real daily cap reached"})

    if source == "real":
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        api_key = os.getenv("GROQ_API_KEY")
    else:
        base_url = os.getenv("STANDIN_BASE_URL", "http://localhost:8000")
        api_key = "dummy"

    client = ModelClient(base_url=base_url, api_key=api_key, model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"))
    scheduler = Scheduler(
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", "4")),
        retry_budget=int(os.getenv("MAX_RETRIES", "2")),
        rpm=int(os.getenv("RPM", "30")),
        tpm=int(os.getenv("TPM", "6000")),
        rpd=int(os.getenv("RPD", "0")),
        give_up_time=float(os.getenv("GIVE_UP_TIME", "30")),
        retries_enabled=os.getenv("DISABLE_RETRIES", "") != "1",
    )
    retriever = _get_retriever(_chunks_for(chunk_size), rerank)
    request_id = f"req-{int(time.time() * 1000)}"

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(etype, payload):
            await queue.put({"event": etype, "data": payload})

        await emit("accepted", {"request_id": request_id, "queue_position": 0})

        scheduler.emit = emit

        async def run():
            try:
                trace = await run_pipeline(
                    question, client, scheduler, retriever, k=k,
                    emit=emit,
                )
                # persist trace
                TRACES[request_id] = trace.to_dict()
                try:
                    Path("results").mkdir(exist_ok=True)
                    trace.save(f"results/trace_{request_id}.json")
                except Exception:
                    pass
                # citation verification from trace? final answer citations
                citations = []
                # parse citations
                import re
                for m in re.finditer(r"\[\s*([^\],]+?)\s*,\s*(\d+)\s*,\s*([^\]]+?)\s*\]", trace.final_answer or ""):
                    citations.append({"paper": m.group(1).strip(), "page": int(m.group(2)), "chunk_id": m.group(3).strip()})
                status = "refused" if trace.refused else "ok"
                if trace.refusal_reason == "unparseable_model_output":
                    status = "error"
                # grade hint
                grade_hint = "refused_correctly" if trace.refused else "correct"
                await emit("final", {
                    "status": status,
                    "answer": trace.final_answer,
                    "citations": citations,
                    "refusal_reason": trace.refusal_reason,
                    "grade_hint": grade_hint,
                    "total_ms": trace.total_ms,
                })
            except DeadlineExceeded as e:
                await emit("final", {"status": "timeout", "answer": str(e), "citations": [], "refusal_reason": str(e), "grade_hint": "timeout", "total_ms": None})
            except Exception as e:
                await emit("final", {"status": "error", "answer": str(e), "citations": [], "refusal_reason": str(e), "grade_hint": "error", "total_ms": None})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
        await task
        await client.aclose()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/chunks/search")
async def search_chunks(q: str = "", size: int = 400):
    path = Path(_chunks_for(size))
    if not path.exists():
        return []
    results = []
    if not q:
        return []
    terms = q.lower().split()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or "").lower()
            score = sum(text.count(t) for t in terms)
            if score > 0:
                results.append({"chunk_id": obj["chunk_id"], "score": score, "paper_id": obj["paper_id"], "page": obj["page"], "section": obj["section"], "text": obj["text"][:300]})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:20]
@app.on_event("startup")
async def preload_retrievers():
    for size in (200, 400, 800):
        for rerank in (False, True):
            _get_retriever(_chunks_for(size), rerank)


_dist = Path("frontend/dist")
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
