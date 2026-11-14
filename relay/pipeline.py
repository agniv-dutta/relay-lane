"""The Relay pipeline: explicit async steps with tracing."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from relay.model_client import ModelClient, RateLimitError
from relay.retrieval import BM25Retriever, Chunk
from relay.scheduler import Scheduler, DailyBudgetExhausted, DeadlineExceeded, TokenBudgetExceeded
from relay.trace import Trace, StepTrace


LANE_MAP = {"understand": 1, "retrieve": 2, "decide": 3, "draft": 4, "verify": 5, "refuse": 0}


DEFAULT_CHUNKS = "data/chunks/chunks_400.jsonl"


# re-exported for callers/tests that import it from here
__all__ = ["TokenBudgetExceeded", "UnparseableOutput", "run_pipeline"]


class UnparseableOutput(Exception):
    pass


def _fit_passages(chunks: list[Chunk], cap_tokens: int) -> tuple[list[Chunk], list[str]]:
    """Drop/truncate passages (lowest score first) until prompt fits cap_tokens."""
    dropped = []
    selected = list(chunks)
    while len(selected) > 1:
        text = _format_passages(selected)
        est = len(text) // 4 + 64
        if est <= cap_tokens:
            break
        d = selected.pop()
        dropped.append(f"dropped {d.chunk_id}")
    text = _format_passages(selected)
    est = len(text) // 4 + 64
    if est > cap_tokens and selected:
        fitted = []
        for c in selected:
            max_chars = max(200, (cap_tokens * 4) // max(1, len(selected)))
            if len(c.text) > max_chars:
                dropped.append(f"truncated {c.chunk_id} to {max_chars} chars")
                newc = Chunk(
                    paper_id=c.paper_id, version=c.version, page=c.page,
                    section=c.section, chunk_id=c.chunk_id,
                    text=c.text[:max_chars] + "...", token_count=c.token_count,
                )
                fitted.append(newc)
            else:
                fitted.append(c)
        selected = fitted
    # final check
    text = _format_passages(selected)
    est = len(text) // 4 + 64
    if est > cap_tokens:
        raise TokenBudgetExceeded(f"prompt {est} tokens exceeds cap {cap_tokens} even after fitting")
    return selected, dropped


def _lane_for(step: str) -> int:
    return LANE_MAP.get(step, 0)


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
    emit=None,
    on_token=None,
    response_format=None,
) -> str:
    step_start_time = time.perf_counter()
    if emit is not None:
        await emit("step_start", {"step": step.name, "lane": _lane_for(step.name)})
    # wrap on_token to emit first_token + token events
    first_emitted = {"done": False}

    async def _on_token(text):
        if not first_emitted["done"]:
            first_emitted["done"] = True
            if emit is not None:
                await emit("first_token", {})
        if emit is not None:
            await emit("token", {"text": text})
        if on_token is not None:
            await on_token(text)

    result = await scheduler.call_model(
        client, messages, step, trace=trace, model=model,
        temperature=temperature, max_tokens=max_tokens, deadline=deadline,
        on_token=_on_token, response_format=response_format,
    )
    step.extra["estimated_tokens"] = len("".join(m.get("content", "") for m in messages)) // 4
    step.extra.setdefault("passages_in_prompt", 0)
    step.extra["actual_tokens_in"] = result.prompt_tokens
    step.extra["actual_tokens_out"] = result.completion_tokens
    step.extra["estimation_error"] = (result.prompt_tokens or 0) - step.extra["estimated_tokens"]
    if "prompt_chars" not in step.extra:
        step.extra["prompt_chars"] = len("".join(m.get("content", "") for m in messages))
    if emit is not None:
        await emit("step_end", {
            "step": step.name,
            "lane": _lane_for(step.name),
            "model_ms": step.model_time_ms,
            "tokens_in": step.prompt_tokens,
            "tokens_out": step.completion_tokens,
        })
    return result.text


async def step_understand(
    scheduler: Scheduler, client: ModelClient, trace: Trace, question: str, model: str | None = None, emit=None
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
    text = await _call_model(scheduler, client, step, trace, messages, temperature=0.0, max_tokens=256, model=model, emit=emit, response_format={"type": "json_object"})
    step.extra["passages_in_prompt"] = 0
    data = _parse_json(text)
    if not data:
        trace.retries += 1
        strict_messages = messages + [{"role": "user", "content": "Return ONLY valid JSON. No prose, no markdown."}]
        text = await _call_model(scheduler, client, step, trace, strict_messages, temperature=0.0, max_tokens=256, model=model, emit=emit, response_format={"type": "json_object"})
        data = _parse_json(text)
        if not data:
            raise UnparseableOutput("understand output unparseable")
    out = {
        "rewritten_query": data.get("rewritten_query") or question,
        "intent": data.get("intent") or "other",
    }
    step.extra["output"] = out
    step.finish()
    return out


async def step_retrieve(
    retriever: BM25Retriever, trace: Trace, query: str, k: int, emit=None
) -> list[Chunk]:
    step = trace.new_step("retrieve")
    if emit is not None:
        await emit("step_start", {"step": "retrieve", "lane": 2})
    t0 = time.perf_counter()
    chunks = await retriever.retrieve(query, k=k)
    step.model_time_ms = 0.0
    step.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    step.extra["query"] = query
    step.extra["k"] = k
    step.extra["num_results"] = len(chunks)
    step.extra["chunk_ids"] = [c.chunk_id for c in chunks]
    step.extra["passages_in_prompt"] = 0
    trace.retrieved_chunks = [c.chunk_id for c in chunks]
    if emit is not None:
        await emit("step_end", {"step": "retrieve", "lane": 2, "model_ms": 0, "tokens_in": None, "tokens_out": None})
    step.finish()
    return chunks


async def step_decide(
    scheduler: Scheduler, client: ModelClient, trace: Trace, question: str, chunks: list[Chunk], model: str | None = None, emit=None, cap_tokens: int = 1500
) -> dict[str, Any]:
    step = trace.new_step("decide")
    fitted, dropped = _fit_passages(chunks, cap_tokens)
    if dropped:
        step.extra["dropped"] = dropped
        if emit is not None:
            for d in dropped:
                await emit("degrade", {"rung": 2, "reason": d})
    passages = _format_passages(fitted)
    step.extra["passages_in_prompt"] = len(fitted)
    step.extra["prompt_chars"] = len(passages) + 100  # approximate
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
    text = await _call_model(scheduler, client, step, trace, messages, temperature=0.0, max_tokens=256, model=model, emit=emit, response_format={"type": "json_object"})
    data = _parse_json(text)
    if not data:
        trace.retries += 1
        strict_messages = messages + [{"role": "user", "content": "Return ONLY valid JSON. No prose."}]
        text = await _call_model(scheduler, client, step, trace, strict_messages, temperature=0.0, max_tokens=256, model=model, emit=emit, response_format={"type": "json_object"})
        data = _parse_json(text)
        if not data:
            raise UnparseableOutput("decide output unparseable")
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
    strict: bool = False, model: str | None = None, max_tokens: int = 768, emit=None, cap_tokens: int = 2000,
) -> str:
    step = trace.new_step("draft")
    fitted, dropped = _fit_passages(chunks, cap_tokens)
    if dropped:
        step.extra["dropped"] = dropped
        if emit is not None:
            for d in dropped:
                await emit("degrade", {"rung": 2, "reason": d})
    passages = _format_passages(fitted)
    step.extra["passages_in_prompt"] = len(fitted)
    step.extra["prompt_chars"] = len(passages) + 100
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
    text = await _call_model(scheduler, client, step, trace, messages, temperature=0.2, max_tokens=max_tokens, model=model, emit=emit)
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
    emit=None,
) -> Trace:
    trace = Trace(question=question)

    # Daily budget exhausted path (clear, no LLM calls)
    if scheduler.daily_exhausted():
        if emit is not None:
            await emit("degrade", {"rung": 5, "reason": "daily budget exhausted"})
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
            if emit is not None:
                await emit("degrade", {"rung": 5, "reason": "load shedding: try later"})
            trace.final_answer = "The service is under heavy load. Please try again later."
            step.finish()
            trace.finish()
            return trace
        trace.decisions.append({"decision": "degradation_level", "level": level, "pressure": round(scheduler.pressure(), 3)})
        if emit is not None:
            await emit("degrade", {"rung": level, "reason": f"pressure {scheduler.pressure():.2f}"})
        u = await step_understand(scheduler, client, trace, question, model=model_for_step, emit=emit)
        query = u["rewritten_query"]

        # degradation: fewer chunks
        k_eff = max(2, k - (2 if level >= 1 else 0))

        # 2-3. retrieve + decide with re-query loop
        chunks: list[Chunk] = []
        decision: dict[str, Any] = {"can_answer": False, "reason": "not attempted", "reformulated_query": None}
        attempts = 0
        while attempts <= max_requery:
            attempts += 1
            chunks = await step_retrieve(retriever, trace, query, k_eff, emit=emit)
            decision = await step_decide(scheduler, client, trace, question, chunks, model=model_for_step, emit=emit)
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
        draft = await step_draft(scheduler, client, trace, question, chunks, model=model_for_draft, max_tokens=max_tok, emit=emit)

        # 5. verify (code-based always; skip any LLM verify pass when degraded)
        step = trace.new_step("verify")
        if emit is not None:
            await emit("step_start", {"step": "verify", "lane": 5})
        ok, problems = verify_citations(draft, chunks)
        step.extra["problems"] = problems
        step.extra["ok"] = ok
        step.extra["degradation_level"] = level
        step.extra["skipped_llm_verify"] = bool(level >= 2)
        if emit is not None:
            await emit("step_end", {"step": "verify", "lane": 5, "model_ms": 0, "tokens_in": None, "tokens_out": None})
        if ok:
            trace.final_answer = draft
            step.finish()
            trace.finish()
            return trace

        # regenerate once (unless degraded level >=2, then refuse directly)
        if level < 2:
            trace.retries += 1
            draft2 = await step_draft(scheduler, client, trace, question, chunks, strict=True, model=model_for_draft, max_tokens=max_tok, emit=emit)
            ok2, problems2 = verify_citations(draft2, chunks)
            step2 = trace.new_step("verify")
            if emit is not None:
                await emit("step_start", {"step": "verify", "lane": 5})
            step2.extra["problems"] = problems2
            step2.extra["ok"] = ok2
            step2.extra["regenerated"] = True
            if emit is not None:
                await emit("step_end", {"step": "verify", "lane": 5, "model_ms": 0, "tokens_in": None, "tokens_out": None})
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
    except RateLimitError as e:
        # attach the trace so callers (check_tokens 429 diagnostics) can read
        # the scheduler's logged decision: bucket state, reserved tokens, reason
        e.trace = trace
        raise
    except TokenBudgetExceeded as e:
        # not a wait: the request cannot fit the token budget even after
        # dropping/truncating passages. Status is cannot_fit_budget, never timeout.
        trace.refused = True
        trace.refusal_reason = "cannot_fit_budget"
        step = trace.new_step("refuse")
        step.extra["reason"] = str(e)
        trace.decisions.append({"decision": "refused", "reason": "cannot_fit_budget", "detail": str(e)})
        trace.final_answer = (
            "The provided papers cannot be used to answer this question within the "
            "token budget: the prompt exceeds the cap even after dropping passages. "
            "Nothing was sent to the model for the failing step."
        )
        step.finish()
        trace.finish()
        return trace
    except UnparseableOutput as e:
        trace.refused = False
        trace.refusal_reason = "unparseable_model_output"
        step = trace.new_step("refuse")
        step.extra["reason"] = "unparseable_model_output"
        trace.final_answer = "error: unparseable_model_output"
        step.finish()
        trace.finish()
        return trace

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
        args.model = os.getenv("MODEL_ID") or os.getenv("MODEL_NAME") or "standin-model"
    if args.base_url is None:
        args.base_url = os.getenv("GROQ_BASE_URL", None)
    if args.api_key is None:
        args.api_key = os.getenv("GROQ_API_KEY", None)

    import asyncio

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
