"""Scheduler regression tests: budget vs wait vs timeout."""

import asyncio
import time

from relay.scheduler import Scheduler, DeadlineExceeded, TokenBudgetExceeded


def test_oversized_estimate_raises_token_budget_not_timeout():
    """Regression: an estimate exceeding capacity must raise TokenBudgetExceeded,
    not be silently clamped and not DeadlineExceeded (it is not a wait)."""
    sched = Scheduler(rpm=30, tpm=6000, rpd=0, give_up_time=0.2)
    try:
        asyncio.run(sched._acquire_tokens(120964, time.time() + 0.2))
    except TokenBudgetExceeded as e:
        assert "token budget exceeded" in str(e)
    except DeadlineExceeded:
        raise AssertionError("oversized estimate must not be DeadlineExceeded")
    else:
        raise AssertionError("expected TokenBudgetExceeded")


def test_no_silent_clamp_of_oversized_estimate():
    # guard clamp removed; oversized estimate now raises instead of being clamped
    sched = Scheduler(rpm=30, tpm=6000, rpd=0, give_up_time=1.0)
    try:
        asyncio.run(sched._acquire_tokens(100000, time.time() + 1.0))
    except TokenBudgetExceeded:
        pass
    except DeadlineExceeded:
        raise AssertionError("oversized estimate must not be DeadlineExceeded")
    else:
        raise AssertionError("expected TokenBudgetExceeded")


def test_tiny_tpm_capacity_raises_budget():
    # estimate fits the request but not the configured bucket capacity
    sched = Scheduler(rpm=30, tpm=500, rpd=0, give_up_time=0.2)
    try:
        asyncio.run(sched._acquire_tokens(1500, time.time() + 0.2))
    except TokenBudgetExceeded as e:
        assert "1500" in str(e) and "500" in str(e)
    else:
        raise AssertionError("expected TokenBudgetExceeded")


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


def test_genuine_wait_past_deadline_is_timeout_not_budget():
    """A real wait that runs past the deadline raises DeadlineExceeded naming
    the gate -- never TokenBudgetExceeded (the request did fit the budget)."""
    sched = Scheduler(rpm=1, tpm=6000, rpd=0, give_up_time=0.05)

    async def run():
        # consume the single rpm token
        await sched._acquire_tokens(1, time.time() + 1.0)
        try:
            await sched._acquire_tokens(1, time.time() + 0.05)
        except DeadlineExceeded as e:
            assert "rpm" in str(e)
            return
        except TokenBudgetExceeded:
            raise AssertionError("a wait past deadline must not be TokenBudgetExceeded")
        raise AssertionError("expected DeadlineExceeded")

    asyncio.run(run())
