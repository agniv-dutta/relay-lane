"""Stand-in model server mimicking Groq's streaming chat-completions API.

Step-aware: reads X-Relay-Step header (or detects from system prompt).
Configurable via JSON file (STANDIN_CONFIG env, default standin/config.json).

Run: uvicorn standin.server:app --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CONFIG_PATH = os.getenv("STANDIN_CONFIG", "standin/config.json")


def load_config() -> dict:
    p = Path(CONFIG_PATH)
    if not p.exists():
        p = Path("standin/config.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


CONFIG = load_config()
app = FastAPI(title="The Relay - Stand-in Model Server (Groq-like)")

# --- state ---
# sliding windows: list of (timestamp, tokens)
_rpm_events: deque = deque()   # (ts, 1) per request
_tpm_events: deque = deque()   # (ts, tokens) per request
_rpd_count = 0
_rpd_day = time.strftime("%Y-%m-%d")
_conc_sem: asyncio.Semaphore | None = None
_in_flight = 0
_in_flight_lock = asyncio.Lock()
_chunk_ids: list[str] = []

_chunks_path = CONFIG.get("content", {}).get("chunks_path", "data/chunks/chunks_400.jsonl")
try:
    with open(_chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _chunk_ids.append(obj["chunk_id"])
            if len(_chunk_ids) >= 50:
                break
except FileNotFoundError:
    _chunk_ids = []


def _cfg(*keys, default=None):
    cur = CONFIG
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _reset_daily():
    global _rpd_count, _rpd_day
    today = time.strftime("%Y-%m-%d")
    if today != _rpd_day:
        _rpd_day = today
        _rpd_count = 0


def _limits():
    L = _cfg("limits", default={})
    return {
        "rpm": int(L.get("rpm", 30)),
        "tpm": int(L.get("tpm", 6000)),
        "rpd": int(L.get("rpd", 0)),
        "max_concurrency": int(L.get("max_concurrency", 4)),
    }


def _sem():
    global _conc_sem
    if _conc_sem is None:
        _conc_sem = asyncio.Semaphore(_limits()["max_concurrency"])
    return _conc_sem


def _prune_windows(now):
    rpm = _limits()["rpm"]
    tpm = _limits()["tpm"]
    while _rpm_events and now - _rpm_events[0] > 60:
        _rpm_events.popleft()
    while _tpm_events and now - _tpm_events[0][0] > 60:
        _tpm_events.popleft()


def _rate_state(now):
    _prune_windows(now)
    _reset_daily()
    rpm_used = len(_rpm_events)
    tpm_used = sum(t for _, t in _tpm_events)
    L = _limits()
    return {
        "rpm_used": rpm_used,
        "rpm_limit": L["rpm"],
        "tpm_used": tpm_used,
        "tpm_limit": L["tpm"],
        "rpd_used": _rpd_count,
        "rpd_limit": L["rpd"],
    }


def _check_limits(est_tokens, now):
    st = _rate_state(now)
    if st["rpd_limit"] > 0 and st["rpd_used"] >= st["rpd_limit"]:
        return False, "rpd", 3600.0
    if st["rpm_used"] >= st["rpm_limit"]:
        if _rpm_events:
            ra = 60 - (now - _rpm_events[0])
        else:
            ra = 1.0
        return False, "rpm", max(0.5, ra)
    if st["tpm_used"] + est_tokens > st["tpm_limit"]:
        if _tpm_events:
            ra = 60 - (now - _tpm_events[0][0])
        else:
            ra = 1.0
        return False, "tpm", max(0.5, ra)
    return True, "", 0.0


def _record_usage(tokens, now):
    global _rpd_count
    _rpm_events.append(now)
    _tpm_events.append((now, tokens))
    _rpd_count += 1


def _reset_time_remaining(window_deque, now):
    if not window_deque:
        return 0
    oldest = window_deque[0] if not isinstance(window_deque[0], tuple) else window_deque[0][0]
    return max(0, 60 - (now - oldest))


def _headers_for(st, now):
    return {
        "x-ratelimit-limit-requests": str(st["rpm_limit"]),
        "x-ratelimit-remaining-requests": str(max(0, st["rpm_limit"] - st["rpm_used"])),
        "x-ratelimit-reset-requests": str(int(_reset_time_remaining(_rpm_events, now))),
        "x-ratelimit-limit-tokens": str(st["tpm_limit"]),
        "x-ratelimit-remaining-tokens": str(max(0, st["tpm_limit"] - st["tpm_used"])),
        "x-ratelimit-reset-tokens": str(int(_reset_time_remaining(_tpm_events, now))),
    }


def _dist_value(spec):
    dist = spec.get("dist", "lognormal")
    if dist == "lognormal":
        mu = float(spec.get("mu", 0.0))
        sigma = float(spec.get("sigma", 1.0))
        return random.lognormvariate(mu, sigma)
    return float(spec.get("mean", 0.1))


def _slowdown(in_flight):
    ls = _cfg("latency", "load_slowdown", default={})
    if not ls.get("enabled", True):
        return 1.0
    factor = float(ls.get("factor_per_request", 0.15))
    cap = float(ls.get("cap", 5.0))
    mult = 1.0 + factor * max(0, in_flight - 1)
    return min(mult, cap)


def _detect_step(request, body):
    step = request.headers.get("x-relay-step", "").strip().lower()
    if step:
        return step
    # detect from system prompt
    for m in body.get("messages", []):
        if m.get("role") == "system":
            content = m.get("content", "").lower()
            if "rewrite and classify" in content or "rewrite" in content:
                return "understand"
            if "judge" in content or "decide if" in content:
                return "decide"
            if "answer the question using only" in content:
                return "draft"
    return "unknown"


def _parse_chunk_ids(text):
    # extract chunk ids from passages like [paper, page, chunk_id]
    return re.findall(r"\[\s*[^\],]+?\s*,\s*\d+\s*,\s*([^\]]+?)\s*\]", text)


def _make_reply(step, messages, body):
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    failures = _cfg("failures", default={})
    # malformed JSON probability
    malformed_rate = float(failures.get("malformed_json_rate", 0.0))
    bad_cite_rate = float(failures.get("bad_citation_rate", 0.0))

    if step == "understand":
        if random.random() < malformed_rate:
            return "this is not json at all"
        return json.dumps({"rewritten_query": last_user[:80], "intent": "method"})

    if step == "decide":
        if random.random() < malformed_rate:
            return "{bad json"
        # sometimes say cannot answer
        if random.random() < 0.3:
            return json.dumps({"can_answer": False, "reason": "insufficient information", "reformulated_query": None})
        return json.dumps({"can_answer": True, "reason": "passages relevant", "reformulated_query": None})

    if step == "draft":
        chunk_ids = _parse_chunk_ids(last_user)
        # pick 1-2 real chunk ids
        cites = []
        if chunk_ids:
            chosen = random.sample(chunk_ids, min(2, len(chunk_ids)))
            cites = chosen
        else:
            cites = _chunk_ids[:2]
        # bad citation injection
        if random.random() < bad_cite_rate and cites:
            cites = cites + ["nonexistent_chunk_999"]
        cite_str = " ".join(f"[{cid.split('_')[0]}, 1, {cid}]" for cid in cites)
        return f"[standin-groq] Based on the papers, regarding '{last_user[:60]}', the attention mechanism computes weighted sums. {cite_str}"

    # default
    return f"[standin-groq] echo: {last_user[:120]}"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    global _in_flight
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "standin-model")
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 256))

    prompt_text = " ".join(m.get("content", "") for m in messages)
    est_prompt_tokens = max(1, len(prompt_text) // 4)
    est_completion_tokens = max(1, max_tokens // 4)
    est_total = est_prompt_tokens + est_completion_tokens

    step = _detect_step(request, body)

    now = time.time()
    allowed, reason, retry_after = _check_limits(est_total, now)
    st = _rate_state(now)
    headers = _headers_for(st, now)

    if not allowed:
        headers["retry-after"] = str(int(retry_after) if retry_after >= 1 else 1)
        return JSONResponse(
            status_code=429,
            content={"error": {"message": f"rate limited ({reason})", "type": "rate_limit_error"}},
            headers=headers,
        )

    sem = _sem()
    await sem.acquire()
    async with _in_flight_lock:
        _in_flight += 1
        in_flight_now = _in_flight

    try:
        failures = _cfg("failures", default={})
        error_rate = float(failures.get("error_rate", 0.0))
        hang_rate = float(failures.get("hang_rate", 0.0))
        slow_rate = float(failures.get("slow_success_rate", 0.0))

        if random.random() < error_rate:
            await asyncio.sleep(0.05)
            return JSONResponse(status_code=500, content={"error": {"message": "internal server error", "type": "server_error"}}, headers=headers)

        hang_seconds = None
        if random.random() < hang_rate:
            hs = failures.get("hang_seconds", [5.0, 30.0])
            hang_seconds = random.uniform(float(hs[0]), float(hs[1]))

        slow_extra = 0.0
        if random.random() < slow_rate:
            se = failures.get("slow_success_extra_ms", [1000.0, 5000.0])
            slow_extra = random.uniform(float(se[0]), float(se[1])) / 1000.0

        ttft_spec = _cfg("latency", "ttft", default={"dist": "lognormal", "mu": -1.6, "sigma": 0.5})
        pt_spec = _cfg("latency", "per_token", default={"dist": "lognormal", "mu": -5.0, "sigma": 0.3})
        mult = _slowdown(in_flight_now)
        ttft = _dist_value(ttft_spec) * mult
        per_token = _dist_value(pt_spec) * mult

        if hang_seconds is not None:
            await asyncio.sleep(hang_seconds)
        await asyncio.sleep(ttft + slow_extra)

        canned = _make_reply(step, messages, body)
        tokens_out = max(1, len(canned.split()))
        actual_total = est_prompt_tokens + tokens_out

        _record_usage(actual_total, time.time())

        st = _rate_state(time.time())
        headers = _headers_for(st, time.time())

        if not stream:
            await asyncio.sleep(per_token * tokens_out)
            return JSONResponse(
                content={
                    "id": "standin-1",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": canned}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": est_prompt_tokens, "completion_tokens": tokens_out, "total_tokens": actual_total},
                },
                headers=headers,
            )

        async def gen():
            for tok in canned.split():
                await asyncio.sleep(per_token)
                chunk = {"id": "standin-1", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id':'standin-1','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':est_prompt_tokens,'completion_tokens':tokens_out,'total_tokens':actual_total}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    finally:
        async with _in_flight_lock:
            _in_flight = max(0, _in_flight - 1)
        sem.release()
PY