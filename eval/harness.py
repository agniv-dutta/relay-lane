"""Eval harness: run questions under multiple retrieval settings."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from relay.model_client import ModelClient
from relay.pipeline import run_pipeline
from relay.retrieval import BM25Retriever, OverlapReranker, NoOpReranker
from relay.scheduler import Scheduler
from eval.rubric import grade as grade_answer

DEFAULT_SETTINGS = [
    {"name": "chunk200_k5_norerank", "chunk_size": 200, "k": 5, "rerank": False},
    {"name": "chunk400_k5_norerank", "chunk_size": 400, "k": 5, "rerank": False},
    {"name": "chunk800_k5_norerank", "chunk_size": 800, "k": 5, "rerank": False},
    {"name": "chunk400_k10_norerank", "chunk_size": 400, "k": 10, "rerank": False},
    {"name": "chunk400_k5_rerank", "chunk_size": 400, "k": 5, "rerank": True},
]

CSV_COLUMNS = [
    "question", "setting", "chunk_size", "k", "rerank", "answer_grade",
    "citation_ok", "refusal_ok", "source_ok", "manual_review", "notes",
    "first_token_ms", "total_ms", "tokens_in", "tokens_out", "retries",
    "final_answer",
]


def chunks_path_for(size: int) -> str:
    return f"data/chunks/chunks_{size}.jsonl"


async def run_one_setting(question: dict, setting: dict, client, scheduler, args) -> dict:
    chunks_path = chunks_path_for(setting["chunk_size"])
    reranker = OverlapReranker() if setting["rerank"] else None
    retriever = BM25Retriever(chunks_path, reranker=reranker)
    # BM25Retriever expects reranker object with rerank method; None -> NoOp? It sets NoOp if None. Pass reranker instance.
    if setting["rerank"]:
        retriever.reranker = OverlapReranker()
    else:
        retriever.reranker = NoOpReranker()

    k = setting["k"]
    trace = await run_pipeline(
        question["question"], client, scheduler, retriever, k=k,
        max_requery=args.max_requery,
        small_model=args.small_model,
        degradation=not args.no_degradation,
    )

    # retrieved chunks as dicts for rubric
    retrieved = []
    # re-retrieve to get chunk dicts? trace only has ids. Load from retriever chunks.
    # We'll map ids to chunks.
    idmap = {c.chunk_id: c for c in retriever.chunks}
    for cid in trace.retrieved_chunks:
        c = idmap.get(cid)
        if c:
            retrieved.append({"chunk_id": c.chunk_id, "paper_id": c.paper_id, "page": c.page})

    final_answer = trace.final_answer
    refused = trace.refused
    g = grade_answer(question, final_answer, refused, retrieved)

    # tokens
    tokens_in = sum(st.prompt_tokens or 0 for st in trace.steps)
    tokens_out = sum(st.completion_tokens or 0 for st in trace.steps)
    ttft = None
    for st in trace.steps:
        if st.first_token_ms is not None:
            ttft = st.first_token_ms
            break

    return {
        "question": question["question"],
        "setting": setting["name"],
        "chunk_size": setting["chunk_size"],
        "k": k,
        "rerank": setting["rerank"],
        "answer_grade": g.answer_grade,
        "citation_ok": g.citation_ok,
        "refusal_ok": g.refusal_ok,
        "source_ok": g.source_ok,
        "manual_review": g.manual_review,
        "notes": "; ".join(g.notes),
        "first_token_ms": ttft if ttft is not None else "",
        "total_ms": trace.total_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retries": trace.retries,
        "final_answer": (final_answer or "").replace("\n", " ")[:500],
    }


async def main_async(args) -> None:
    # load questions
    with open(args.questions, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    if not isinstance(questions, list):
        questions = questions.get("questions", [])

    # load settings
    if args.settings:
        with open(args.settings, encoding="utf-8") as f:
            settings = json.load(f)
    else:
        settings = DEFAULT_SETTINGS

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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    results = []
    for setting in settings:
        for q in questions:
            print(f"[eval] setting={setting['name']} q={q['question'][:40]}")
            try:
                row = await run_one_setting(q, setting, client, scheduler, args)
            except Exception as e:
                row = {
                    "question": q["question"],
                    "setting": setting["name"],
                    "chunk_size": setting["chunk_size"],
                    "k": setting["k"],
                    "rerank": setting["rerank"],
                    "answer_grade": "error",
                    "citation_ok": False,
                    "refusal_ok": False,
                    "source_ok": False,
                    "manual_review": True,
                    "notes": str(e),
                    "first_token_ms": "",
                    "total_ms": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "retries": 0,
                    "final_answer": "",
                }
            writer.writerow(row)
            f.flush()
            results.append(row)

    f.close()
    await client.aclose()

    # summary
    summary = {}
    for setting in settings:
        rows = [r for r in results if r["setting"] == setting["name"]]
        n = len(rows)
        correct = sum(1 for r in rows if r["answer_grade"] in ("correct", "refused_correctly"))
        summary[setting["name"]] = {
            "n": n,
            "accuracy": round(correct / n, 3) if n else 0,
            "avg_ttft_ms": round(sum((r["first_token_ms"] or 0) for r in rows) / n, 1) if n else 0,
            "avg_tokens_out": round(sum(r["tokens_out"] for r in rows) / n, 1) if n else 0,
            "avg_tokens_in": round(sum(r["tokens_in"] for r in rows) / n, 1) if n else 0,
            "manual_review_count": sum(1 for r in rows if r["manual_review"]),
        }
    summary_path = out_path.with_name("eval_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults: {out_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--settings", default=None, help="JSON file with list of settings")
    parser.add_argument("--out", default="results/eval_results.csv")
    parser.add_argument("--source", choices=["real", "stand-in"], default="stand-in")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--max-requery", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--tpm", type=int, default=6000)
    parser.add_argument("--rpd", type=int, default=0)
    parser.add_argument("--give-up-time", type=float, default=60.0)
    parser.add_argument("--disable-retries", action="store_true")
    parser.add_argument("--small-model", default=None)
    parser.add_argument("--no-degradation", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        args.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    if args.source == "real" and not args.api_key and not os.getenv("GROQ_API_KEY"):
        print("WARNING: no GROQ_API_KEY set")

    import asyncio
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
