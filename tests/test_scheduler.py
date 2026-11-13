"""Scheduler regression tests."""

import asyncio
import time

from relay.scheduler import Scheduler, DeadlineExceeded


def test_oversized_estimate_raises_token_budget():
    """Regression: an estimate exceeding capacity must raise, not be silently clamped."""
    sched = Scheduler(rpm=30, tpm=6000, rpd=0, give_up_time=0.2)
    try:
        asyncio.run(sched._acquire_tokens(120964, time.time() + 0.2))
    except DeadlineExceeded as e:
        assert "token budget exceeded" in str(e)
    else:
        raise AssertionError("expected DeadlineExceeded")


def test_guard_clamps_and_logs():
    # guard clamp removed; oversized estimate now raises
    sched = Scheduler(rpm=30, tpm=6000, rpd=0, give_up_time=1.0)
    try:
        asyncio.run(sched._acquire_tokens(100000, time.time() + 1.0))
    except DeadlineExceeded:
        pass
    else:
        raise AssertionError("expected DeadlineExceeded")


def test_timeout_names_gate():
    sched = Scheduler(rpm=1, tpm=6000, rpd=0, give_up_time=0.05)

    async def run():
        await sched._acquire_tokens(1, time.time() + 1.0)
        try:
            await sched._acquire_tokens(1, time.time() + 0.05)
        except DeadlineExceeded as e:
            assert "rpm" in str(e)
        else:
            raise AssertionError("expected DeadlineExceeded")

    asyncio.run(run())
