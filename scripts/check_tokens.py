"""Print a per-step token table for N questions through the real pipeline.

Usage:
    python scripts/check_tokens.py --n 5 --setting chunk400-k5 --base-url http://127.0.0.1:8001/openai/v1

Prints one row per pipeline step (understand, retrieve, decide, draft, verify)
plus any retry/regenerate rows, totals per question, the average, and:
    questions/min by TPM, by RPM, which binds, questions/day by TPD.
If any 429 happens, prints who sent it, the stand-in's reason, the limiter's
bucket state at send time, and the stand-in's live counters.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import httpx

from relay.model_client import ModelClient, ModelEndpointUnreachable, RateLimitError
from relay.pipeline import run_pipeline
from relay.retrieval import BM25Retriever, NoOpReranker
from relay.scheduler import Scheduler


def load_limits(base_url: str) -> dict:
    """rpm/tpm/rpd from stand-in config (these are stand-in settings, not Groq's)."""
    try:
        cfg = json.loads(Path("standin/config.json").read_text(encoding="utf-8"))
        L = cfg.get("limits", {})
        return {
            "rpm": L.get("rpm"),
            "tpm": L.get("tpm"),
            "rpd": L.get("rpd"),
            "tpd": L.get("tpd"),
        }
    except Exception:
        return {"rpm": None, "tpm": None, "rpd": None, "tpd": None}


from pathlib import Path


async def standin_counters(base_url: str) -> dict:
    stats_url = base_url.rsplit("/openai/v1", 1)[0] + "/openai/v1/stats"
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(stats_url)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {"error": f"could not fetch {stats_url}"}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--setting", default="chunk400-k5")
    parser.add_argument("--base-url", default=os.getenv("STANDIN_BASE_URL", "http://127.0.0.1:8001/openai/v1"))
    parser.add_argument("--questions", default="data/questions.json")
    args = parser.parse_args()

    m = re.search(r"chunk(\d+)", args.setting)
    chunk_size = int(m.group(1)) if m else 400
    m = re.search(r"[-_]k(\d+)", args.setting)
    k = int(m.group(1)) if m else 5

    qs = json.load(open(args.questions, encoding="utf-8"))[: args.n]

    limits = load_limits(args.base_url)
    rpm_limit = limits["rpm"] or 30
    tpm_limit = limits["tpm"] or 6000

    client = ModelClient(base_url=args.base_url, api_key="dummy", model="standin-model")
    scheduler = Scheduler(
        max_concurrency=4, rpm=rpm_limit, tpm=tpm_limit, rpd=limits["rpd"] or 0,
        give_up_time=60, retry_budget=1,
    )
    retriever = BM25Retriever(f"data/chunks/chunks_{chunk_size}.jsonl", reranker=NoOpReranker())

    print(f"base_url={args.base_url}  setting={args.setting}")
    print(f"limits (stand-in config): rpm={rpm_limit} tpm={tpm_limit} rpd={limits['rpd']} tpd={limits['tpd']}\n")
    hdr = f"{'qid':<5}{'step':<11}{'lane':>4}{'passages':>9}{'prompt_chars':>13}{'est_in':>8}{'actual_in':>10}{'actual_out':>11}{'status':>9}"
    print(hdr)
    print("-" * len(hdr))

    total_tokens = 0
    total_calls = 0
    n_fatal = 0
    all_429_decisions: list[dict] = []

    for qi, q in enumerate(qs, 1):
        qid = f"q{qi}"
        try:
            trace = await run_pipeline(q["question"], client, scheduler, retriever, k=k)
        except RateLimitError as e:
            print(f"{qid:<5}{'FATAL':<11}{'':>4}{'':>9}{'':>13}{'':>8}{'':>10}{'':>11}{'429':>9}  {str(e)[:80]}")
            n_fatal += 1
            # the pipeline attaches the trace so the scheduler's 429 decision
            # (bucket state, reserved tokens) survives the exception
            tr = getattr(e, "trace", None)
            if tr is not None:
                for d in tr.decisions:
                    if d.get("decision") in ("failed", "retry") and d.get("status") == 429:
                        d2 = dict(d)
                        d2["qid"] = qid
                        all_429_decisions.append(d2)
            continue
        except ModelEndpointUnreachable as e:
            print(f"{qid:<5}{'FATAL':<11}  model_endpoint_unreachable: {e.detail}")
            break
        q_tokens = 0
        q_calls = 0
        for st in trace.steps:
            lane = {"understand": 1, "retrieve": 2, "decide": 3, "draft": 4, "verify": 5}.get(st.name, 0)
            is_model_call = st.name in ("understand", "decide", "draft") or st.prompt_tokens is not None
            if not is_model_call and st.name not in ("retrieve", "verify"):
                continue
            est = st.extra.get("estimated_tokens", "")
            pc = st.extra.get("prompt_chars", "")
            pip = st.extra.get("passages_in_prompt", 0)
            if st.prompt_tokens is not None:
                status = "ok"
            elif st.name in ("retrieve", "verify"):
                status = "code"
            else:
                status = "n/a"
            print(f"{qid:<5}{st.name:<11}{lane:>4}{pip:>9}{str(pc):>13}{str(est):>8}{str(st.prompt_tokens or ''):>10}{str(st.completion_tokens or ''):>11}{status:>9}")
            if is_model_call:
                q_tokens += (st.prompt_tokens or 0) + (st.completion_tokens or 0)
                q_calls += 1
            # retries: extra attempt rows
            attempts = st.extra.get("attempts") or 1
            if attempts > 1:
                for a in range(2, attempts + 1):
                    print(f"{qid:<5}{st.name + '/retry' + str(a - 1):<11}{lane:>4}{'':>9}{'':>13}{'':>8}{'':>10}{'':>11}{'retry':>9}")
                    total_calls += 1
        # regenerate rows (draft run twice)
        n_drafts = sum(1 for st in trace.steps if st.name == "draft")
        if n_drafts > 1:
            for i in range(2, n_drafts + 1):
                print(f"{qid:<5}{'draft/regen':<11}{4:>4}")
        per_status = "refused" if trace.refused else "ok"
        print(f"{qid:<5}{'TOTAL':<11}{'':>4}{'':>9}{'':>13}{'':>8}{'':>10}{q_tokens:>11}{per_status:>9}")
        total_tokens += q_tokens
        total_calls += q_calls
        # collect 429 diagnostics from the trace (retries that later succeeded)
        for d in trace.decisions:
            if d.get("decision") in ("failed", "retry") and d.get("status") == 429:
                d2 = dict(d)
                d2["qid"] = qid
                all_429_decisions.append(d2)

    n = len(qs)
    avg = total_tokens / n if n else 0
    avg_calls = total_calls / n if n else 0
    print(f"\nquestions run: {n}")
    print(f"total tokens (in+out): {total_tokens}")
    print(f"avg tokens/question: {avg:.1f}")
    print(f"avg model calls/question: {avg_calls:.1f}")
    print(f"\nquestions/min by TPM = {tpm_limit} / {avg:.1f} = {(tpm_limit / avg) if avg else float('nan'):.1f}")
    print(f"questions/min by RPM = {rpm_limit} / {avg_calls:.1f} = {(rpm_limit / avg_calls) if avg_calls else float('nan'):.1f}")
    if avg and avg_calls:
        qpm_tpm = tpm_limit / avg
        qpm_rpm = rpm_limit / avg_calls
        binds = "TPM" if qpm_tpm < qpm_rpm else "RPM"
        print(f"binding: {binds}")
    tpd = limits["tpd"]
    if tpd and avg:
        print(f"questions/day by TPD = {tpd} / {avg:.1f} = {tpd / avg:.1f} (from config)")
    else:
        print("questions/day by TPD: not measured (no tpd in stand-in config)")

    n_429 = len(all_429_decisions)
    if all_429_decisions:
        n_terminal = sum(1 for d in all_429_decisions if d.get("decision") == "failed")
        n_retry = n_429 - n_terminal
        print(f"\n=== 429 diagnostics ({n_429} 429s: {n_terminal} terminal, {n_retry} retried) ===")
        for d in all_429_decisions:
            if d.get("decision") == "retry":
                print(f"\nquestion {d['qid']}: 429 on step {d.get('step', '?')} (attempt {d.get('attempt')}, reserved {d.get('reserved')} tokens) - retried after {d.get('backoff_s')}s")
                continue
            print(f"\nquestion {d['qid']}: 429 on step {d.get('step', '?')} - no retry left (reserved {d.get('reserved')} tokens)")
            print(f"  stand-in reason: {d.get('standin_reason', '?')}")
            print(f"  limiter bucket state at send: rpm {d.get('rpm_bucket')}/{d.get('rpm_capacity')}, tpm {d.get('tpm_bucket')}/{d.get('tpm_capacity')}, in_flight {d.get('in_flight')}")
            counters = await standin_counters(args.base_url)
            print(f"  stand-in counters now: {counters}")
            if d.get("rpm_bucket") is not None and d.get("rpm_capacity"):
                print(f"  why admitted: limiter's rpm bucket had {d.get('rpm_bucket')}/{d.get('rpm_capacity')} and tpm {d.get('tpm_bucket')}/{d.get('tpm_capacity')} tokens left")
                print("  (limiter's token bucket refills continuously; stand-in counts a sliding 60s window including output tokens)")
    elif n_fatal:
        print(f"\n{n_fatal} request(s) ended in 429; no per-decision detail captured.")
    else:
        print("\nno 429 in this run")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
