"""Trace object for recording per-step pipeline metrics."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepTrace:
    name: str
    start: float = 0.0
    end: float = 0.0
    duration_ms: float = 0.0
    queue_wait_ms: float = 0.0
    model_time_ms: float = 0.0
    retry_time_ms: float = 0.0
    first_token_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status: str = "ok"
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "ok") -> None:
        self.end = time.perf_counter()
        self.duration_ms = round((self.end - self.start) * 1000, 2)
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "duration_ms": self.duration_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "model_time_ms": self.model_time_ms,
            "retry_time_ms": self.retry_time_ms,
            "first_token_ms": self.first_token_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "status": self.status,
            "extra": self.extra,
        }


class Trace:
    def __init__(self, question: str) -> None:
        self.question = question
        self.start = time.perf_counter()
        self.end: float = 0.0
        self.steps: list[StepTrace] = []
        self.decisions: list[dict[str, Any]] = []
        self.final_answer: str | None = None
        self.refused: bool = False
        self.refusal_reason: str | None = None
        self.retrieved_chunks: list[str] = []
        self.retries: int = 0

    def new_step(self, name: str) -> StepTrace:
        st = StepTrace(name=name, start=time.perf_counter())
        self.steps.append(st)
        return st

    def finish(self) -> None:
        self.end = time.perf_counter()

    @property
    def total_ms(self) -> float:
        return round((self.end - self.start) * 1000, 2) if self.end else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "total_ms": self.total_ms,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "final_answer": self.final_answer,
            "retrieved_chunks": self.retrieved_chunks,
            "retries": self.retries,
            "decisions": self.decisions,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return p
