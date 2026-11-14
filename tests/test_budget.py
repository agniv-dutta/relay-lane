"""Budget behavior: degrade (drop passages) -> fit, else refuse cannot_fit_budget.

cannot_fit_budget is never reported as timeout.
"""

import asyncio
import json

from relay.model_client import ModelResponse
from relay.pipeline import _fit_passages, _format_passages, run_pipeline
from relay.retrieval import Chunk
from relay.scheduler import Scheduler, DeadlineExceeded


def make_chunks(n, size=400):
    return [
        Chunk(paper_id="p", version="v", page=i, section="s", chunk_id=f"c{i}", text="x " * size)
        for i in range(n)
    ]


def test_degradation_drops_passages_until_fits():
    """Many passages over the cap: drop lowest-score first, no exception."""
    chunks = make_chunks(20, size=400)
    try:
        fitted, dropped = _fit_passages(chunks, 500)
    except Exception as e:
        raise AssertionError(f"degradation must fit by dropping, not raise: {e!r}")
    assert dropped, "expected passages to be dropped"
    assert len(fitted) < len(chunks)
    est = len(_format_passages(fitted)) // 4 + 64
    assert est <= 500, f"fitted prompt {est} > cap 500"


class _StubClient:
    """Returns canned step-aware output; never raises."""

    async def stream(self, messages, temperature=0.0, max_tokens=512, model_override=None,
                     on_token=None, response_format=None, step=None):
        system = messages[0]["content"]
        if "rewrite and classify" in system:
            text = json.dumps({"rewritten_query": "q", "intent": "other"})
        elif "judge" in system:
            text = json.dumps({"can_answer": True, "reason": "ok", "reformulated_query": None})
        else:
            text = "Answer [p, 1, c0]"
        if on_token is not None:
            await on_token(text)
        return ModelResponse(
            text=text, first_token_ms=1.0, total_ms=2.0,
            prompt_tokens=10, completion_tokens=5, headers={}, status=200,
        )

    async def aclose(self):
        pass


class _HugeChunkRetriever:
    """Returns one passage that cannot fit the decide cap (1500 tokens)."""

    async def retrieve(self, query, k=5):
        return [Chunk(paper_id="p", version="v", page=1, section="s",
                      chunk_id="huge", text="x " * 10000)]


def test_pipeline_refuses_cannot_fit_budget_not_timeout():
    """Even one passage over the cap after fitting -> refusal_reason
    cannot_fit_budget, never DeadlineExceeded/timeout."""
    sched = Scheduler(rpm=30, tpm=6000, rpd=0, give_up_time=5.0)
    try:
        trace = asyncio.run(
            run_pipeline("what is attention?", _StubClient(), sched, _HugeChunkRetriever(), k=1)
        )
    except DeadlineExceeded:
        raise AssertionError("cannot fit the budget must not surface as DeadlineExceeded")
    assert trace.refused is True
    assert trace.refusal_reason == "cannot_fit_budget", trace.refusal_reason
    assert trace.final_answer
    assert any(s.name == "refuse" for s in trace.steps)
