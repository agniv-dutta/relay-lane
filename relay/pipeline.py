"""The Relay pipeline: explicit async steps with tracing."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from relay.model_client import ModelClient
from relay.retrieval import BM25Retriever, Chunk
from relay.scheduler import Scheduler, DailyBudgetExhausted, DeadlineExceeded
from relay.trace import Trace, StepTrace


DEFAULT_CHUNKS = "data/chunks/chunks_400.jsonl"


def _format_passages(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[{c.paper_id}, {c.page}, {c.chunk_id}]\n"
            f"section: {c.section}\n"
            f"text: {c.text}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # try to extract a JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def _call_model(
    scheduler: Scheduler,
    client: ModelClient,
    step: StepTrace,
    trace: Trace,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: str | None = None,
    deadline: float | None = None,
) -> str:
    result = await scheduler.call_model(
        client, messages, step, trace=trace, model=model,
        temperature=temperature, max_tokens=max_tokens, deadline=deadline,
    )
    return result.text


async def step_understand(
    scheduler: Scheduler, client: ModelClient, trace: Trace, question: str, model: str | None = None
) -> dict[str, str]:
    step = trace.new_step("understand")
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite and classify questions about three arXiv papers. "
                "Return JSON only: {\"rewritten_query\": str, \"intent\": str}."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]
    text = await _call_model(scheduler, client, step, trace, messages, temperature=0.0, max_tokens=256, model=model)
    data = _parse_json(text)
    data = _parse_json(text)
    out = {
        "rewritten_query": data.get("rewritten_query") or question,
        "intent": data.get("intent") or "other",
    }
    step.extra["output"] = out
    step.finish()
    return out


async def step_retrieve(
    retriever: BM25Retriever, trace: Trace, query: str, k: int
) -> list[Chunk]:
    step = trace.new_step("retrieve")
    t0 = time.perf_counter()
    chunks = await retriever.retrieve(query, k=k)
    step.model_time_ms = 0.0
    step.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    step.extra["query"] = query
    step.extra["k"] = k
    step.extra["num_results"] = len(chunks)
    step.extra["chunk_ids"] = [c.chunk_id for c in chunks]
    trace.retrieved_chunks = [c.chunk_id for c in chunks]
    step.finish()
    return chunks


async def step_decide(
    scheduler: Scheduler, client: ModelClient, trace: Trace, question: str, chunks: list[Chunk], model: str | None = None
) -> dict[str, Any]:
    step = trace.new_step("decide")
    passages = _format_passages(chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a judge. Decide if the provided passages can answer the question. "
                "Return JSON only: {\"can_answer\": bool, \"reason\": str, "
                "\"reformulated_query\": str or null}."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nPassages:\n{passages}",
        },
    ]
    text = await _call_model(scheduler, client, step, trace, messages, temperature=0.0, max_tokens=256, model=model)
    data = _parse_json(text)
    out = {
        "can_answer": bool(data.get("can_answer", False)),
        "reason": data.get("reason", "insufficient information"),
        "reformulated_query": data.get("reformulated_query"),
    }
    step.extra["output"] = out
    step.finish()
    return out


async def step_draft(
    scheduler: Scheduler, client: ModelClient, trace: Trace, question: str, chunks: list[Chunk],
    strict: bool = False, model: str | None = None, max_tokens: int = 768,
) -> str:
    step = trace.new_step("draft")
    passages = _format_passages(chunks)
    strict_note = (
        " Be extra careful: cite EVERY claim; never invent citations. "
        if strict
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using ONLY the provided passages. "
                "Cite every factual claim as [paper_id, page, chunk_id]. "
                "Never fabricate citations. If unsupported, omit the claim." + strict_note
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nPassages:\n{passages}",
        },
    ]
    text = await _call_model(scheduler, client, step, trace, messages, temperature=0.2, max_tokens=max_tokens, model=model)
    step.extra["draft_preview"] = text[:300]
    step.finish()
    return text


CITATION_RE = re.compile(r"\[\s*([^\],]+?)\s*,\s*(\d+)\s*,\s*([^\]]+?)\s*\]")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_citations(draft: str, chunks: list[Chunk]) -> tuple[bool, list[str]]:
    """Code-based verification. Returns (ok, list of problems)."""
    problems: list[str] = []
    chunk_map = {c.chunk_id: c for c in chunks}
    # also index by (paper_id, page)
    by_key = {(c.paper_id, c.page, c.chunk_id): c for c in chunks}

    citations = CITATION_RE.findall(draft)
    if not citations:
        # allow very short refusals? but draft should have citations for facts
        if len(draft.split()) > 20:
            problems.append("no citations found in draft")
        return (len(problems) == 0), problems

    for paper_id, page_str, chunk_id in citations:
        paper_id = paper_id.strip()
        chunk_id = chunk_id.strip()
        page = int(page_str)
        c = chunk_map.get(chunk_id)
        if c is None:
            problems.append(f"unknown chunk_id: {chunk_id}")
            continue
        if c.paper_id != paper_id or c.page != page:
            problems.append(
                f"citation mismatch: [{paper_id}, {page}, {chunk_id}] "
                f"!= actual [{c.paper_id}, {c.page}, {c.chunk_id}]"
            )

    # quote check: any double-quoted phrase must appear in retrieved text
    blob = "\n".join(c.text for c in chunks)
    blob_norm = _normalize(blob)
    for m in re.finditer(r'"([^"]{8,})"', draft):
        quote = _normalize(m.group(1))
        if quote and quote not in blob_norm:
            problems.append(f"quoted phrase not found in retrieved chunks: {m.group(1)[:60]}...")

    return (len(problems) == 0), problems


def _daily_exhausted(trace: Trace) -> Trace:
    trace.refused = True
    trace.refusal_reason = "daily budget exhausted"
    step = trace.new_step("refuse")
    step.extra["reason"] = "daily budget exhausted"
    trace.decisions.append({"decision": "daily_budget_exhausted", "reason": "RPD limit reached"})
    trace.final_answer = "The daily request budget is exhausted. Please try again later."
    step.finish()
    trace.finish()
    return trace


async def run_pipeline(
    question: str,
    client: ModelClient,
    scheduler: Scheduler,
    retriever: BM25Retriever,
    k: int = 5,
    max_requery: int = 2,
    small_model: str | None = None,
    degradation: bool = True,
) -> Trace:
    trace = Trace(question=question)

    # Daily budget exhausted path (clear, no LLM calls)
    if scheduler.daily_exhausted():
        return _daily_exhausted(trace)

    try:
        def deg_level() -> int:
            if not degradation:
                return 0
            p = scheduler.pressure()
            if p >= 0.98 or scheduler.daily_exhausted():
                return 4
            if p >= 0.9:
                return 3
            if p >= 0.8:
                return 2
            if p >= 0.6:
                return 1
            return 0

        # 1. understand
        level = deg_level()
        model_for_step = small_model if (level >= 3 and small_model) else None
        if level >= 4:
            trace.refused = True
            trace.refusal_reason = "load shedding: try later"
            step = trace.new_step("refuse")
            step.extra["reason"] = "load shedding level 4"
            trace.decisions.append({"decision": "shed", "level": level, "reason": "near limit"})
            trace.final_answer = "The service is under heavy load. Please try again later."
            step.finish()
            trace.finish()
            return trace
        trace.decisions.append({"decision": "degradation_level", "level": level, "pressure": round(scheduler.pressure(), 3)})
        u = await step_understand(scheduler, client, trace, question, model=model_for_step)
        query = u["rewritten_query"]

        # degradation: fewer chunks
        k_eff = max(2, k - (2 if level >= 1 else 0))

        # 2-3. retrieve + decide with re-query loop
        chunks: list[Chunk] = []
        decision: dict[str, Any] = {"can_answer": False, "reason": "not attempted", "reformulated_query": None}
        attempts = 0
        while attempts <= max_requery:
            attempts += 1
            chunks = await step_retrieve(retriever, trace, query, k_eff)
            decision = await step_decide(scheduler, client, trace, question, chunks, model=model_for_step)
            if decision["can_answer"]:
                break
            trace.retries += 1
            if decision.get("reformulated_query"):
                query = decision["reformulated_query"]
            else:
                break
            if attempts > max_requery:
                break

        if not decision["can_answer"]:
            trace.refused = True
            trace.refusal_reason = decision["reason"]
            step = trace.new_step("refuse")
            step.extra["reason"] = decision["reason"]
            msg = (
                "The provided papers do not cover this question sufficiently. "
                f"Specifically: {decision['reason']}. "
                "I cannot answer without fabricating citations."
            )
            trace.final_answer = msg
            step.finish()
            trace.finish()
            return trace

        # 4. draft
        level = deg_level()
        model_for_draft = small_model if (level >= 3 and small_model) else None
        max_tok = 512 if level >= 2 else 768
        draft = await step_draft(scheduler, client, trace, question, chunks, model=model_for_draft, max_tokens=max_tok)

        # 5. verify (code-based always; skip any LLM verify pass when degraded)
        step = trace.new_step("verify")
        ok, problems = verify_citations(draft, chunks)
        step.extra["problems"] = problems
        step.extra["ok"] = ok
        step.extra["degradation_level"] = level
        step.extra["skipped_llm_verify"] = bool(level >= 2)
        if ok:
            trace.final_answer = draft
            step.finish()
            trace.finish()
            return trace

        # regenerate once (unless degraded level >=2, then refuse directly)
        if level < 2:
            trace.retries += 1
            draft2 = await step_draft(scheduler, client, trace, question, chunks, strict=True, model=model_for_draft, max_tokens=max_tok)
            ok2, problems2 = verify_citations(draft2, chunks)
            step2 = trace.new_step("verify")
            step2.extra["problems"] = problems2
            step2.extra["ok"] = ok2
            step2.extra["regenerated"] = True
            if ok2:
                trace.final_answer = draft2
                step2.finish()
                trace.finish()
                return trace
            problems = problems2

        # refuse
        trace.refused = True
        trace.refusal_reason = "verification failed: " + "; ".join(problems)
        step.finish()
        refuse_step = trace.new_step("refuse")
        refuse_step.extra["reason"] = trace.refusal_reason
        trace.final_answer = (
            "The provided papers do not cover this question sufficiently to produce a "
            "verified answer. " + trace.refusal_reason
        )
        refuse_step.finish()
        trace.finish()
        return trace


    except DailyBudgetExhausted:
        return _daily_exhausted(trace)

async def main_async(args) -> None:
    client = ModelClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    scheduler = Scheduler(
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        rpm=args.rpm,
        tpm=args.tpm,
        rpd=args.rpd,
        give_up_time=args.give_up_time,
        retries_enabled=not args.disable_retries,
    )
    retriever = BM25Retriever(args.chunks, reranker=None)
    try:
        trace = await run_pipeline(
            args.question, client, scheduler, retriever, k=args.k, max_requery=args.max_requery,
            small_model=args.small_model, degradation=not args.no_degradation,
        )
    finally:
        await client.aclose()

    print("\n=== RESULT ===")
    print(trace.final_answer)
    print("\n=== TRACE ===")
    print(json.dumps(trace.to_dict(), ensure_ascii=False, indent=2)[:4000])
    if args.trace_out:
        p = trace.save(args.trace_out)
        print(f"\nTrace saved: {p}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (Groq or stand-in)")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-requery", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--tpm", type=int, default=6000)
    parser.add_argument("--rpd", type=int, default=0)
    parser.add_argument("--give-up-time", type=float, default=30.0)
    parser.add_argument("--disable-retries", action="store_true")
    parser.add_argument("--small-model", default=None)
    parser.add_argument("--no-degradation", action="store_true")
    parser.add_argument("--trace-out", default=None)
    args = parser.parse_args()

    if args.model is None:
        args.model = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    if args.base_url is None:
        args.base_url = os.getenv("GROQ_BASE_URL", None)
    if args.api_key is None:
        args.api_key = os.getenv("GROQ_API_KEY", None)

    import asyncio

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
