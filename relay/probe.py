"""Probe Groq API (OpenAI-compatible, streaming) for latency, rate limits, 429s, concurrency.

Usage:
    python -m relay.probe --model llama-3.1-8b-instant --n 5 --max-requests 20
    python -m relay.probe --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
RESULTS_DIR = Path("results")
RAW_LOG = RESULTS_DIR / "probe_raw.jsonl"

SHORT_PROMPT = "What is 2+2? Reply with a single number."
LONG_PROMPT = (
    "Summarize the following passage in exactly 3 bullet points.\n\n"
    + ("The quick brown fox jumps over the lazy dog. " * 60)
)

_state_lock = asyncio.Lock()
_state = {"used": 0, "cap": 0}


def _load_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set (check .env)")
    return key


def rate_limit_headers(resp: httpx.Response) -> dict:
    out = {}
    for k, v in resp.headers.items():
        lk = k.lower()
        if lk.startswith("x-ratelimit-") or lk == "retry-after":
            out[lk] = v
    return out


def _log_record(rec: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))
    return s[idx]


async def _reserve_slot() -> bool:
    async with _state_lock:
        if _state["used"] >= _state["cap"]:
            return False
        _state["used"] += 1
        return True


async def stream_request(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    prompt_type: str,
    mode: str,
    concurrency_level: int | None = None,
    attempt: int = 0,
    extra: dict | None = None,
) -> dict | None:
    if not await _reserve_slot():
        print("[cap] total request cap reached; skipping")
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 256 if prompt_type == "short" else 512,
    }

    rec = {
        "ts": time.time(),
        "mode": mode,
        "model": model,
        "prompt_type": prompt_type,
        "prompt_chars": len(prompt),
        "attempt": attempt,
        "concurrency_level": concurrency_level,
        "status": None,
        "ttft_s": None,
        "total_s": None,
        "rate_limit_headers": {},
        "usage": None,
        "completion_preview": None,
        "body_preview": None,
        "error": None,
        "recovery_s": None,
    }
    if extra:
        rec.update(extra)

    start = time.perf_counter()
    first_token = None
    collected = []
    status = None
    headers = {}
    body_text = ""
    usage = None
    error = None

    try:
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            status = resp.status_code
            headers = rate_limit_headers(resp)
            if status >= 400:
                await resp.aread()
                body_text = resp.text
            else:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token is None:
                                first_token = time.perf_counter()
                            collected.append(content)
        end = time.perf_counter()
    except Exception as e:
        end = time.perf_counter()
        error = repr(e)

    rec["status"] = status
    rec["rate_limit_headers"] = headers
    rec["usage"] = usage
    rec["total_s"] = round(end - start, 4)
    rec["ttft_s"] = round(first_token - start, 4) if first_token is not None else None
    if collected:
        rec["completion_preview"] = "".join(collected)[:300]
    if body_text:
        rec["body_preview"] = body_text[:1000]
    if error:
        rec["error"] = error

    _log_record(rec)
    # console summary
    rl = headers.get("x-ratelimit-remaining-requests") or headers.get("x-ratelimit-remaining-tokens")
    print(
        f"[{mode}] {prompt_type} attempt={attempt} status={status} "
        f"ttft={rec['ttft_s']} total={rec['total_s']} rl_rem={rl}"
    )
    return rec


async def run_sequential(client, model, n, cap):
    print(f"\n=== SEQUENTIAL: n={n} per prompt type, cap={cap} ===")
    for prompt_type, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
        for i in range(n):
            rec = await stream_request(client, model, prompt, prompt_type, "sequential", attempt=i)
            if rec is None:
                return
            # 429 recovery handling
            if rec.get("status") == 429:
                retry_after = None
                try:
                    retry_after = float(rec["rate_limit_headers"].get("retry-after", "2"))
                except (ValueError, TypeError):
                    retry_after = 2.0
                retry_after = max(0.5, min(retry_after, 30.0))
                print(f"  429 received; waiting {retry_after}s to recover...")
                t0 = time.perf_counter()
                await asyncio.sleep(retry_after)
                # retry once
                r2 = await stream_request(
                    client, model, prompt, prompt_type, "sequential",
                    attempt=i, extra={"is_retry_after_429": True},
                )
                if r2 is not None:
                    r2["recovery_s"] = round(time.perf_counter() - t0, 4)
                    # re-log recovery time by appending? We'll just print.
                    print(f"  recovery attempt status={r2.get('status')} recovery_s={r2['recovery_s']}")


async def run_concurrent(client, model, max_level, cap):
    print(f"\n=== CONCURRENT RAMP: up to {max_level}, cap={cap} ===")
    prev_p95 = None
    for level in range(1, max_level + 1):
        # check cap
        async with _state_lock:
            if _state["used"] + level > cap:
                print(f"[cap] not enough remaining for level {level}; stopping")
                break
        tasks = [
            stream_request(client, model, SHORT_PROMPT, "short", "concurrent",
                           concurrency_level=level, attempt=i)
            for i in range(level)
        ]
        results = await asyncio.gather(*tasks)
        oks = [r for r in results if r and r.get("status") == 200]
        errs = [r for r in results if r and r.get("status") != 200]
        lats = [r["total_s"] for r in oks if r.get("total_s") is not None]
        ttfts = [r["ttft_s"] for r in oks if r.get("ttft_s") is not None]
        p95 = _percentile(lats, 95)
        print(
            f"  level={level}: ok={len(oks)}/{level} err={len(errs)} "
            f"p50_lat={_percentile(lats,50)} p95_lat={p95} "
            f"p50_ttft={_percentile(ttfts,50)}"
        )
        # stop conditions
        if errs:
            print("  errors observed; stopping ramp")
            break
        if prev_p95 is not None and p95 is not None and p95 > prev_p95 * 2:
            print("  significant slowdown; stopping ramp")
            break
        prev_p95 = p95
        await asyncio.sleep(1.0)


async def main_async(args):
    global _state
    _state["cap"] = args.max_requests
    _state["used"] = 0

    if args.dry_run:
        seq_reqs = 2 * args.n * 2  # short+long, with a retry budget
        conc_reqs = sum(range(1, args.concurrency_max + 1))
        print("DRY RUN - no API calls")
        print(f"  model: {args.model}")
        print(f"  sequential: {args.n} per prompt type x2 types, plus 429 retries (budgeted)")
        print(f"  estimated sequential requests (incl retry budget): ~{seq_reqs}")
        print(f"  concurrent ramp: levels 1..{args.concurrency_max} => {conc_reqs} requests")
        print(f"  hard cap (max_requests): {args.max_requests}")
        print(f"  raw log: {RAW_LOG}")
        return

    api_key = _load_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(base_url=GROQ_BASE, headers=headers, timeout=120.0) as client:
        if args.mode in ("sequential", "all"):
            await run_sequential(client, args.model, args.n, args.max_requests)
        if args.mode in ("concurrent", "all"):
            await run_concurrent(client, args.model, args.concurrency_max, args.max_requests)

    print(f"\nDone. Total requests used: {_state['used']}/{_state['cap']}")
    print(f"Raw log: {RAW_LOG}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=5, help="sequential requests per prompt type")
    parser.add_argument("--max-requests", type=int, default=20, help="HARD CAP on total API calls")
    parser.add_argument("--mode", choices=["sequential", "concurrent", "all"], default="all")
    parser.add_argument("--concurrency-max", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_requests < 1:
        raise SystemExit("--max-requests must be >= 1")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
