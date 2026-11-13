"""Rate-limit-aware scheduler with token buckets, deadlines, retries, shedding."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from relay.model_client import RateLimitError, HTTPError
from relay.trace import Trace


class DeadlineExceeded(Exception):
    pass


class DailyBudgetExhausted(Exception):
    pass


@dataclass
class TokenBucket:
    rate_per_sec: float
    capacity: float
    tokens: float = field(init=False)
    last: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last = time.time()

    def refill(self, now: float) -> None:
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        self.last = now


@dataclass
class DailyCounter:
    limit: int
    used: int = 0
    day: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    def _reset_if_needed(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.used = 0

    def increment(self, n: int = 1) -> None:
        self._reset_if_needed()
        self.used += n

    def exhausted(self) -> bool:
        self._reset_if_needed()
        return self.limit > 0 and self.used >= self.limit


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    text = " ".join(m.get("content", "") for m in messages)
    return max(1, len(text) // 4)


class Scheduler:
    def __init__(
        self,
        max_concurrency: int = 4,
        rpm: int = 30,
        tpm: int = 6000,
        rpd: int = 0,
        give_up_time: float = 30.0,
        retry_budget: int = 3,
        retries_enabled: bool = True,
        base_backoff: float = 1.0,
        backoff_cap: float = 30.0,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.sem = asyncio.Semaphore(max_concurrency)
        self.rpm_bucket = TokenBucket(rate_per_sec=rpm / 60.0, capacity=float(rpm))
        self.tpm_bucket = TokenBucket(rate_per_sec=tpm / 60.0, capacity=float(tpm))
        self.daily = DailyCounter(limit=rpd)
        self.daily_tokens = 0
        self.give_up_time = give_up_time
        self.retry_budget = retry_budget
        self.retries_enabled = retries_enabled
        self.base_backoff = base_backoff
        self.backoff_cap = backoff_cap
        self.emit = None  # optional async callable(event_type, payload)
        self._in_flight = 0
        self._in_flight_lock = asyncio.Lock()
        self.decisions: list[dict[str, Any]] = []

    def pressure(self) -> float:
        now = time.time()
        self.rpm_bucket.refill(now)
        self.tpm_bucket.refill(now)
        r1 = 1.0 - (self.rpm_bucket.tokens / max(1.0, self.rpm_bucket.capacity))
        r2 = 1.0 - (self.tpm_bucket.tokens / max(1.0, self.tpm_bucket.capacity))
        daily_r = (self.daily.used / self.daily.limit) if self.daily.limit > 0 else 0.0
        conc_r = self._in_flight / max(1, self.max_concurrency)
        return max(r1, r2, daily_r, conc_r)

    def daily_exhausted(self) -> bool:
        return self.daily.exhausted()

    async def _acquire_tokens(self, est: int, deadline: float, trace=None) -> None:
        # If the estimate exceeds bucket capacity, do not clamp silently.
        # Raise so the caller can degrade (shrink passages) or refuse; the
        # reserved estimate must equal what is sent.
        if est > self.tpm_bucket.capacity and self.tpm_bucket.capacity > 0:
            raise DeadlineExceeded(
                f"token budget exceeded: estimate {est} > tpm capacity {self.tpm_bucket.capacity:.0f}"
            )
        if 1 > self.rpm_bucket.capacity and self.rpm_bucket.capacity > 0:
            raise DeadlineExceeded(
                f"token budget exceeded: rpm estimate 1 > rpm capacity {self.rpm_bucket.capacity:.0f}"
            )

        while True:
            now = time.time()
            if now >= deadline:
                # Determine which gate blocked.
                self.rpm_bucket.refill(now)
                self.tpm_bucket.refill(now)
                if self.rpm_bucket.tokens < 1:
                    gate = "rpm"
                    detail = f"rpm tokens={self.rpm_bucket.tokens:.1f}/{self.rpm_bucket.capacity:.0f}"
                elif self.tpm_bucket.tokens < est:
                    gate = "tpm"
                    detail = f"tpm tokens={self.tpm_bucket.tokens:.1f}/{self.tpm_bucket.capacity:.0f} est={est}"
                else:
                    gate = "concurrency"
                    detail = f"in_flight={self._in_flight}/{self.max_concurrency}"
                raise DeadlineExceeded(f"token bucket wait exceeded deadline: blocked by {gate} ({detail})")
            self.rpm_bucket.refill(now)
            self.tpm_bucket.refill(now)
            if self.rpm_bucket.tokens >= 1 and self.tpm_bucket.tokens >= est:
                self.rpm_bucket.tokens -= 1
                self.tpm_bucket.tokens -= est
                self.daily.increment(1)
                self.daily_tokens += est
                return
            await asyncio.sleep(min(0.05, deadline - now))

    def _correct_tokens(self, est: int, actual: int) -> None:
        diff = actual - est
        if diff == 0:
            return
        now = time.time()
        self.tpm_bucket.refill(now)
        if diff > 0:
            self.tpm_bucket.tokens = max(0.0, self.tpm_bucket.tokens - diff)
        else:
            self.tpm_bucket.tokens = min(self.tpm_bucket.capacity, self.tpm_bucket.tokens - diff)

    def _backoff(self, attempt: int, retry_after: float | None, deadline: float) -> float:
        if retry_after is not None:
            base = float(retry_after)
        else:
            base = min(self.backoff_cap, self.base_backoff * (2 ** (attempt - 1)))
            base = random.uniform(0, base)
        remaining = deadline - time.time()
        return max(0.0, min(base, remaining))

    async def call_model(
        self,
        client: Any,
        messages: list[dict[str, str]],
        step: Any,
        trace: Trace | None = None,
        priority: int = 0,
        estimated_tokens: int | None = None,
        deadline: float | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        on_token=None,
        response_format=None,
    ) -> Any:
        est = estimated_tokens if estimated_tokens is not None else estimate_tokens(messages)
        if deadline is None:
            deadline = time.time() + self.give_up_time

        step.queue_wait_ms = 0.0
        step.retry_time_ms = 0.0

        if self.daily.exhausted():
            decision = {"decision": "daily_budget_exhausted", "reason": "RPD limit reached"}
            if trace is not None:
                trace.decisions.append(decision)
            step.extra["scheduler_decision"] = decision
            raise DailyBudgetExhausted("daily budget exhausted")

        # concurrency acquire with deadline
        queue_start = time.time()
        acquired = False
        while time.time() < deadline:
            try:
                await asyncio.wait_for(self.sem.acquire(), timeout=min(0.1, deadline - time.time()))
                acquired = True
                break
            except asyncio.TimeoutError:
                continue
        if not acquired:
            decision = {"decision": "dropped", "reason": "concurrency/deadline exceeded while waiting"}
            if trace is not None:
                trace.decisions.append(decision)
            step.extra["scheduler_decision"] = decision
            raise DeadlineExceeded("gave up waiting for concurrency slot")

        async with self._in_flight_lock:
            self._in_flight += 1
        queue_wait = time.time() - queue_start
        step.queue_wait_ms = round(queue_wait * 1000, 2)

        decision_log: dict[str, Any] = {"decision": "admitted", "queue_wait_ms": step.queue_wait_ms, "priority": priority}
        if trace is not None:
            trace.decisions.append(decision_log)
        step.extra["scheduler_decision"] = decision_log

        retry_time = 0.0
        attempts = 0
        try:
            await self._acquire_tokens(est, deadline, trace=trace)

            while True:
                if time.time() >= deadline:
                    raise DeadlineExceeded("deadline exceeded before attempt")
                attempts += 1
                t0 = time.time()
                try:
                    resp = await client.stream(messages, temperature=temperature, max_tokens=max_tokens, model_override=model, on_token=on_token, response_format=response_format)
                    step.model_time_ms = round((time.time() - t0) * 1000, 2)
                    step.first_token_ms = resp.first_token_ms
                    step.prompt_tokens = resp.prompt_tokens
                    step.completion_tokens = resp.completion_tokens
                    step.extra["attempts"] = attempts
                    step.extra["retry_time_ms"] = round(retry_time * 1000, 2)
                    actual = (resp.prompt_tokens or 0) + (resp.completion_tokens or 0)
                    if actual > 0:
                        self._correct_tokens(est, actual)
                    return resp
                except RateLimitError as e:
                    if not self.retries_enabled or attempts > self.retry_budget or time.time() >= deadline:
                        decision = {"decision": "failed", "reason": "rate limited, no retry", "status": 429}
                        if trace is not None:
                            trace.decisions.append(decision)
                        if self.emit is not None:
                            await self.emit("retry", {"attempt": attempts, "backoff_ms": 0, "reason": "rate limited, no retry"})
                        step.extra["scheduler_decision"] = decision
                        raise
                    backoff = self._backoff(attempts, e.retry_after, deadline)
                    await asyncio.sleep(backoff)
                    retry_time += backoff
                    decision = {"decision": "retry", "reason": "429 rate limited", "backoff_s": round(backoff, 3), "attempt": attempts}
                    if trace is not None:
                        trace.decisions.append(decision)
                    if self.emit is not None:
                        await self.emit("retry", {"attempt": attempts, "backoff_ms": int(backoff * 1000), "reason": "429 rate limited"})
                except HTTPError as e:
                    if e.status < 500:
                        raise
                    if not self.retries_enabled or attempts > self.retry_budget or time.time() >= deadline:
                        decision = {"decision": "failed", "reason": f"server error {e.status}, no retry"}
                        if trace is not None:
                            trace.decisions.append(decision)
                        step.extra["scheduler_decision"] = decision
                        raise
                    backoff = self._backoff(attempts, None, deadline)
                    await asyncio.sleep(backoff)
                    retry_time += backoff
                    decision = {"decision": "retry", "reason": f"5xx {e.status}", "backoff_s": round(backoff, 3), "attempt": attempts}
                    if trace is not None:
                        trace.decisions.append(decision)
                    if self.emit is not None:
                        await self.emit("retry", {"attempt": attempts, "backoff_ms": int(backoff * 1000), "reason": f"5xx {e.status}"})
                except Exception:
                    raise
        finally:
            async with self._in_flight_lock:
                self._in_flight = max(0, self._in_flight - 1)
            self.sem.release()
            step.extra["retry_time_ms"] = round(retry_time * 1000, 2)
            step.retry_time_ms = round(retry_time * 1000, 2)
