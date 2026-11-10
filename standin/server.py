"""Stand-in model server mimicking Groq's streaming chat-completions API.

Configurable via JSON file (STANDIN_CONFIG env, default standin/config.json).

Run: uvicorn standin.server:app --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CONFIG_PATH = os.getenv("STANDIN_CONFIG", "standin/config.json")


def load_config() -> dict:
    p = Path(CONFIG_PATH)
    if not p.exists():
        p = Path("config.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


CONFIG = load_config()
app = FastAPI(title="The Relay - Stand-in Model Server (Groq-like)")

# --- state ---
_rpm_window: deque[float] = deque()
_tpm_window: deque[tuple[float, int]] = deque()
_rpd_count = 0
_rpd_day = time.strftime("%Y-%m-%d")
_conc_sem: asyncio.Semaphore | None = None
_in_flight = 0
_in_flight_lock = asyncio.Lock()
_chunk_ids: list[str] = []

# load chunk ids for citations
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


def _sem() -> asyncio.Semaphore:
    global _conc_sem
    if _conc_sem is None:
        _conc_sem = asyncio.Semaphore(_limits()["max_concurrency"])
    return _conc_sem


def _dist_value(spec: dict) -> float:
    dist = spec.get("dist", "lognormal")
    if dist == "lognormal":
        mu = float(spec.get("mu", 0.0))
        sigma = float(spec.get("sigma", 1.0))
        return random.lognormvariate(mu, sigma)
    # fallback
    return float(spec.get("mean", 0.1))


def _rate_state() -> dict:
    now = time.time()
    rpm = _limits()["rpm"]
    tpm = _limits()["tpm"]
    # prune rpm window (60s)
    while _rpm_window and now - _rpm_window[0] > 60:
        _rpm_window.popleft()
    # prune tpm window
    while _tpm_window and now - _tpm_window[0][0] > 60:
        _tpm_window.popleft()
    used_rpm = len(_rpm_window)
    used_tpm = sum(t for _, t in _tpm_window)
    _reset_daily()
    return {
        "rpm_used": used_rpm,
        "rpm_limit": rpm,
        "tpm_used": used_tpm,
        "tpm_limit": tpm,
        "rpd_used": _rpd_count,
        "rpd_limit": _limits()["rpd"],
    }


def _check_limits(est_tokens: int) -> tuple[bool, str, float]:
    """Return (allowed, reason, retry_after)."""
    st = _rate_state()
    # concurrency checked separately via semaphore
    if st["rpd_limit"] > 0 and st["rpd_used"] >= st["rpd_limit"]:
        return False, "rpd", 3600.0
    if st["rpm_used"] >= st["rpm_limit"]:
        # retry after until oldest request expires
        if _rpm_window:
            ra = 60 - (time.time() - _rpm_window[0])
        else:
            ra = 1.0
        return False, "rpm", max(0.5, ra)
    if st["tpm_used"] + est_tokens > st["tpm_limit"]:
        if _tpm_window:
            ra = 60 - (time.time() - _tpm_window[0][0])
        else:
            ra = 1.0
        return False, "tpm", max(0.5, ra)
    return True, "", 0.0


def _record_usage(tokens: int):
    global _rpd_count
    _rpm_window.append(time.time())
    _tpm_window.append((time.time(), tokens))
    _rpd_count += 1


def _slowdown(in_flight: int) -> float:
    ls = _cfg("latency", "load_slowdown", default={})
    if not ls.get("enabled", True):
        return 1.0
    factor = float(ls.get("factor_per_request", 0.15))
    cap = float(ls.get("cap", 5.0))
    mult = 1.0 + factor * max(0, in_flight - 1)
    return min(mult, cap)


def _make_canned_reply(messages: list) -> str:
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    # extract chunk ids mentioned in the prompt (so verify step can pass)
    import re
    found = re.findall(r"\[\s*([^\],]+?)\s*,\s*(\d+)\s*,\s*([^\]]+?)\s*\]", last_user)
    cites = []
    seen = set()
    for paper, page, cid in found:
        cid = cid.strip()
        if cid not in seen:
            seen.add(cid)
            cites.append((paper.strip(), page.strip(), cid))
        if len(cites) >= 3:
            break
    if not cites and _chunk_ids:
        for cid in _chunk_ids[:3]:
            paper = cid.split("_")[0]
            cites.append((paper, "1", cid))
    cite_str = " ".join(f"[{p}, {pg}, {c}]" for p, pg, c in cites)
    return (
        f"[standin-groq] Based on the provided papers, regarding '{last_user[:80]}', "
        f"the attention mechanism computes weighted sums of values. {cite_str}"
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _in_flight
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "standin-model")
    stream = bool(body.get("stream", False))
    max_tokens = int(body.get("max_tokens", 256))

    # estimate tokens
    prompt_text = " ".join(m.get("content", "") for m in messages)
    est_prompt_tokens = max(1, len(prompt_text) // 4)
    est_completion_tokens = max(1, max_tokens // 4)
    est_total = est_prompt_tokens + est_completion_tokens

    # rate limit check
    allowed, reason, retry_after = _check_limits(est_total)
    if not allowed:
        _limits_now = _rate_state()
        headers = {
            "x-ratelimit-limit-requests": str(_limits_now["rpm_limit"]),
            "x-ratelimit-remaining-requests": str(max(0, _limits_now["rpm_limit"] - _limits_now["rpm_used"])),
            "x-ratelimit-limit-tokens": str(_limits_now["tpm_limit"]),
            "x-ratelimit-remaining-tokens": str(max(0, _limits_now["tpm_limit"] - _limits_now["tpm_used"])),
            "retry-after": str(int(retry_after) if retry_after >= 1 else 1),
        }
        return JSONResponse(
            status_code=429,
            content={"error": {"message": f"rate limited ({reason})", "type": "rate_limit_error"}},
            headers=headers,
        )

    # concurrency gate
    sem = _sem()
    # acquire with timeout? if at capacity, queue (realistic). We'll just acquire.
    await sem.acquire()
    async with _in_flight_lock:
        _in_flight += 1
        in_flight_now = _in_flight

    try:
        # failure injection
        failures = _cfg("failures", default={})
        error_rate = float(failures.get("error_rate", 0.0))
        hang_rate = float(failures.get("hang_rate", 0.0))
        slow_rate = float(failures.get("slow_success_rate", 0.0))

        if random.random() < error_rate:
            # random error
            await asyncio.sleep(0.05)
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "internal server error", "type": "server_error"}},
            )

        hang_seconds = None
        if random.random() < hang_rate:
            hs = failures.get("hang_seconds", [5.0, 30.0])
            hang_seconds = random.uniform(float(hs[0]), float(hs[1]))

        slow_extra = 0.0
        if random.random() < slow_rate:
            se = failures.get("slow_success_extra_ms", [1000.0, 5000.0])
            slow_extra = random.uniform(float(se[0]), float(se[1])) / 1000.0

        # latency draws
        ttft_spec = _cfg("latency", "ttft", default={"dist": "lognormal", "mu": -1.6, "sigma": 0.5})
        pt_spec = _cfg("latency", "per_token", default={"dist": "lognormal", "mu": -5.0, "sigma": 0.3})
        mult = _slowdown(in_flight_now)
        ttft = _dist_value(ttft_spec) * mult
        per_token = _dist_value(pt_spec) * mult

        if hang_seconds is not None:
            await asyncio.sleep(hang_seconds)
        await asyncio.sleep(ttft + slow_extra)

        canned = _make_canned_reply(messages)
        tokens = canned.split()

        # record usage
        _record_usage(est_total)

        st = _rate_state()
        headers = {
            "x-ratelimit-limit-requests": str(st["rpm_limit"]),
            "x-ratelimit-remaining-requests": str(max(0, st["rpm_limit"] - st["rpm_used"])),
            "x-ratelimit-limit-tokens": str(st["tpm_limit"]),
            "x-ratelimit-remaining-tokens": str(max(0, st["tpm_limit"] - st["tpm_used"])),
        }

        if not stream:
            # non-streaming: sleep for all tokens
            await asyncio.sleep(per_token * len(tokens))
            return JSONResponse(
                content={
                    "id": "standin-1",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": canned}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": est_prompt_tokens, "completion_tokens": len(tokens), "total_tokens": est_prompt_tokens + len(tokens)},
                },
                headers=headers,
            )

        async def gen():
            for tok in tokens:
                await asyncio.sleep(per_token)
                chunk = {
                    "id": "standin-1",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": tok + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'id':'standin-1','object':'chat.completion.chunk','model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'usage':{'prompt_tokens':est_prompt_tokens,'completion_tokens':len(tokens),'total_tokens':est_prompt_tokens+len(tokens)}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    finally:
        async with _in_flight_lock:
            _in_flight = max(0, _in_flight - 1)
        sem.release()
