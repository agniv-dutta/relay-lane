"""Print per-model-call token table for N questions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

from dotenv import load_dotenv

load_dotenv()

from relay.model_client import ModelClient
from relay.pipeline import run_pipeline
from relay.retrieval import BM25Retriever, NoOpReranker
from relay.scheduler import Scheduler


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--setting", default="chunk400-k5")
    parser.add_argument("--base-url", default=os.getenv("STANDIN_BASE_URL", "http://localhost:8001"))
    parser.add_argument("--questions", default="data/questions.json")
    args = parser.parse_args()

    # parse setting like chunk400-k5 or chunk400_k5_norerank
    import re
    m = re.search(r"chunk(\d+)", args.setting)
    chunk_size = int(m.group(1)) if m else 400
    m = re.search(r"k(\d+)", args.setting)
    k = int(m.group(1)) if m else 5

    import json
    qs = json.load(open(args.questions, encoding="utf-8"))
    qs = qs[: args.n]

    client = ModelClient(base_url=args.base_url, api_key="dummy", model="m")
    scheduler = Scheduler(max_concurrency=4, rpm=30, tpm=6000, rpd=0, give_up_time=60, retry_budget=1)
    retriever = BM25Retriever(f"data/chunks/chunks_{chunk_size}.jsonl", reranker=NoOpReranker())

    print(f"{'qid':<6}{'step':<12}{'passages':>9}{'prompt_chars':>13}{'est_in':>8}{'actual_in':>10}{'actual_out':>11}{'status':>8}")

    total_tokens = 0
    total_calls = 0
    per_q_tokens = []
    for qi, q in enumerate(qs, 1):
        qid = f"q{qi}"
        trace = await run_pipeline(q["question"], client, scheduler, retriever, k=k)
        q_tokens = 0
        q_calls = 0
        for st in trace.steps:
            if st.extra.get("estimated_tokens") is None and st.prompt_tokens is None:
                continue
            # only model-call steps
            if st.name in ("understand", "decide", "draft", "verify"):
                est = st.extra.get("estimated_tokens", "")
                pc = st.extra.get("prompt_chars", "")
                pip = st.extra.get("passages_in_prompt", 0)
                status = "ok" if st.prompt_tokens is not None else "n/a"
                print(f"{qid:<6}{st.name:<12}{pip:>9}{pc:>13}{est:>8}{str(st.prompt_tokens or ''):>10}{str(st.completion_tokens or ''):>11}{status:>8}")
                q_tokens += (st.prompt_tokens or 0) + (st.completion_tokens or 0)
                q_calls += 1
        per_q_tokens.append(q_tokens)
        print(f"{qid:<6}{'TOTAL':<12}{'':>9}{'':>13}{'':>8}{'':>10}{q_tokens:>11}{'':>8}")
        total_tokens += q_tokens

    n = len(qs)
    avg = total_tokens / n if n else 0
    avg_calls = 3  # understand, decide, draft (verify is code-only)
    print(f"\navg tokens/question: {avg:.1f}")
    print(f"questions/min by TPM = 6000 / {avg:.1f} = {6000/avg:.1f}")
    print(f"questions/min by RPM = 30 / {avg_calls} = {30/avg_calls:.1f}")
    print(f"binding: {'TPM' if 6000/avg < 30/avg_calls else 'RPM'}")

    await client.aclose()


asyncio.run(main())
PY