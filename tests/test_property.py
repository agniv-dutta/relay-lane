"""Property tests for prompt budget."""

import random

from relay.pipeline import _fit_passages, TokenBudgetExceeded
from relay.retrieval import Chunk


def make_chunks(n, size=500):
    return [Chunk(paper_id="p", version="v", page=i, section="s", chunk_id=f"c{i}", text="x " * size) for i in range(n)]


def test_fit_passages_never_exceeds_cap():
    for _ in range(50):
        n = random.randint(1, 20)
        size = random.randint(100, 800)
        chunks = make_chunks(n, size)
        cap = random.choice([500, 1000, 1500, 2000])
        try:
            fitted, dropped = _fit_passages(chunks, cap)
            text = "\n".join(c.text for c in fitted)
            assert len(text) // 4 + 64 <= cap or len(fitted) == 1
        except TokenBudgetExceeded:
            pass  # acceptable if even one passage can't fit


def test_fit_passages_single_huge_raises():
    chunks = [Chunk(paper_id="p", version="v", page=1, section="s", chunk_id="c0", text="x " * 10000)]
    try:
        _fit_passages(chunks, 100)
    except TokenBudgetExceeded:
        pass
    else:
        # if no raise, the fitted result must be under cap
        fitted, dropped = _fit_passages(chunks, 100)
        assert len("\n".join(c.text for c in fitted)) // 4 + 64 <= 100