"""Load generator: open-loop arrivals over named scenarios.

Usage:
    python -m loadgen.run --scenario steady --source stand-in --rate 1 --duration 20
    python -m loadgen.run --scenario burst --source stand-in --rate 2 --duration 30 --burst-size 10
    python -m loadgen.run --scenario sustained_overload --source stand-in --rate 10 --duration 20
    python -m loadgen.run --scenario rpm_halved --source stand-in --rate 2 --duration 20 --rpm-factor 0.5
    python -m loadgen.run --scenario daily_exhausted --source stand-in --rate 5 --duration 20 --rpd 1
    python -m loadgen.run --scenario steady --source real --rate 0.5 --duration 60 --max-requests 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from relay.model_client import ModelClient
from relay.pipeline import run_pipeline
from relay.retrieval import BM25Retriever
from relay.scheduler import Scheduler

SCENARIOS = ["steady", "burst", "sustained_overload", "rpm_halved", "daily_exhausted"]

CSV_COLUMNS = [
    "request_id", "scenario", "source", "model_id", "run_date",
    "start_time", "first_token_ms", "total_ms",
    "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
]


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def grade_answer(q: dict, final_answer: str | None, refused: bool) -> str:
    answerable = q.get("answerable", True)
    if refused:
        return "refused_correctly" if not answerable else "refused_wrongly"
    # answered
    if not answerable:
        return "wrong"
    expected = (q.get("expected") or "").lower()
    ans = (final_answer or "").lower()
    if expected and expected in ans:
        return "correct"
    # accept if any expected token present
    if expected and any(tok in ans for tok in expected.split() if len(tok) > 3):
        return "correct"
    return "wrong"


def arrival_times(scenario: str, rate: float, duration: float, burst_size: int) -> list[float]:
    """Open-loop arrival schedule (seconds from start)."""
    times = []
    if scenario == "burst":
        for _ in range(burst_size):
            times.append(0.0)
        # then steady
        t = 0.0
        while t < duration:
            t += random.expovariate(rate) if rate > 0 else duration + 1
            if t < duration:
                times.append(t)
    elif scenario == "sustained_overload":
        t = 0.0
        r = rate * 3  # overload
        while t < duration:
            t += random.expovariate(r) if r > 0 else duration + 1
            if t < duration:
                times.append(t)
    else:
        t = 0.0
        while t < duration:
            t += random.expovariate(rate) if rate > 0 else duration + 1
            if t < duration:
                times.append(t)
    times.sort()
    return times


async def run_one(
    req_id: str,
    scenario: str,
    source: str,
    question: dict,
    args,
    client,
    scheduler,
    retriever,
    k: int,
    csv_writer,
    csv_lock: asyncio.Lock,
    start_offset: float,
) -> None:
    await asyncio.sleep(start_offset)
    start_time = time.time()
    first_token_ms = None
    total_ms = None
    status = "ok"
    retries = 0
    tokens_in = 0
    tokens_out = 0
    final_answer = None
    refused = False
    try:
        trace = await asyncio.wait_for(
            run_pipeline(
                question["question"], client, scheduler, retriever, k=k,
                max_requery=args.max_requery,
                small_model=args.small_model,
                degradation=not args.no_degradation,
            ),
            timeout=args.request_timeout,
        )
        retries = trace.retries
        final_answer = trace.final_answer
        refused = trace.refused
        total_ms = trace.total_ms
        # first token ms: first non-null among steps
        for st in trace.steps:
            if st.first_token_ms is not None:
                first_token_ms = st.first_token_ms
                break
        # tokens
        for st in trace.steps:
            if st.prompt_tokens:
                tokens_in += st.prompt_tokens
            if st.completion_tokens:
                tokens_out += st.completion_tokens
    except asyncio.TimeoutError:
        status = "timeout"
    except Exception as e:
        import traceback; traceback.print_exc()
        name = type(e).__name__
        if "RateLimit" in name or "DailyBudget" in name:
            status = "rate_limited"
        elif "Deadline" in name or "Timeout" in name:
            status = "timeout"
        else:
            status = "error"
        # try to extract retries? no
    if status == "ok":
        # status already ok; but if refused? still ok (completed)
        pass

    grade = grade_answer(question, final_answer, refused) if status == "ok" else ""

    row = {
        "request_id": req_id,
        "scenario": scenario,
        "source": source,
        # metadata: which model produced this row, and when
        "model_id": args.model,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "start_time": round(start_time, 3),
        "first_token_ms": first_token_ms if first_token_ms is not None else "",
        "total_ms": total_ms if total_ms is not None else "",
        "status": status,
        "retries": retries,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retrieval_setting": f"k={k}",
        "answer_grade": grade,
    }
    async with csv_lock:
        csv_writer.writerow(row)
        # flush
        csv_writer._file.flush() if hasattr(csv_writer, "_file") else None


async def main_async(args) -> None:
    source = args.source
    if source == "real":
        base_url = args.base_url or os.getenv("GROQ_BASE_URL")
        api_key = args.api_key or os.getenv("GROQ_API_KEY")
    else:
        base_url = args.base_url or os.getenv("STANDIN_BASE_URL", "http://localhost:8000")
        api_key = args.api_key or "dummy"

    client = ModelClient(base_url=base_url, api_key=api_key, model=args.model)
    scheduler = Scheduler(
        max_concurrency=args.max_concurrency,
        retry_budget=args.max_retries,
        rpm=args.rpm,
        tpm=args.tpm,
        rpd=args.rpd,
        give_up_time=args.give_up_time,
        retries_enabled=not args.disable_retries,
    )
    retriever = BM25Retriever(args.chunks, reranker=None)
    questions = load_questions(args.questions)

    # hard cap for real
    max_requests = args.max_requests
    if source == "real" and max_requests is None:
        max_requests = 20
    if max_requests is None:
        max_requests = 10**9

    times = arrival_times(args.scenario, args.rate, args.duration, args.burst_size)
    # apply max_requests cap
    if len(times) > max_requests:
        times = times[:max_requests]

    # open output csv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    # attach file for flush
    writer._file = f
    lock = asyncio.Lock()

    k = args.k
    tasks = []
    for i, t in enumerate(times):
        q = questions[i % len(questions)]
        req_id = f"{args.scenario}-{i+1:05d}"
        task = asyncio.create_task(run_one(
            req_id, args.scenario, source, q, args, client, scheduler, retriever, k,
            writer, lock, t,
        ))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)
    f.close()
    await client.aclose()
    print(f"Done. {len(times)} requests -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--source", choices=["real", "stand-in"], default="stand-in")
    parser.add_argument("--rate", type=float, default=1.0, help="arrival rate (req/s)")
    parser.add_argument("--duration", type=float, default=30.0, help="duration seconds")
    parser.add_argument("--burst-size", type=int, default=10)
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--chunks", default="data/chunks/chunks_400.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-requery", type=int, default=2)
    parser.add_argument("--out", default="results/results.csv")
    parser.add_argument("--max-requests", type=int, default=None, help="hard cap (required-ish for real)")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--tpm", type=int, default=6000)
    parser.add_argument("--rpd", type=int, default=0)
    parser.add_argument("--give-up-time", type=float, default=30.0)
    parser.add_argument("--disable-retries", action="store_true")
    parser.add_argument("--small-model", default=None)
    parser.add_argument("--no-degradation", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.getenv("MODEL_ID") or os.getenv("MODEL_NAME") or "standin-model"
    if args.source == "real" and args.max_requests is None:
        args.max_requests = 20
        print(f"[real source] defaulting --max-requests to {args.max_requests}")

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
