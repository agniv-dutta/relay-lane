"""Pluggable model client (OpenAI-compatible, streaming)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class HTTPError(Exception):
    def __init__(self, status: int, body: str = ""):
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body[:200]}")


class RateLimitError(HTTPError):
    def __init__(self, status: int = 429, retry_after: float | None = None, body: str = ""):
        super().__init__(status, body)
        self.retry_after = retry_after


@dataclass
class ModelResponse:
    text: str
    first_token_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    headers: dict[str, str]
    status: int


class ModelClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "dummy")
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        model_override: str | None = None,
        on_token=None,
        response_format=None,
    ) -> ModelResponse:
        model = model_override or self.model
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        start = time.perf_counter()
        first_token = None
        collected: list[str] = []
        usage: dict[str, Any] | None = None
        headers: dict[str, str] = {}
        status = 0
        body_text = ""
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
                status = resp.status_code
                headers = {k.lower(): v for k, v in resp.headers.items()}
                if status == 429:
                    await resp.aread()
                    body_text = resp.text
                    ra = headers.get("retry-after")
                    retry_after = float(ra) if ra else None
                    raise RateLimitError(status=status, retry_after=retry_after, body=body_text)
                if status >= 400:
                    await resp.aread()
                    body_text = resp.text
                    raise HTTPError(status=status, body=body_text)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token is None:
                                first_token = time.perf_counter()
                            collected.append(content)
                            if on_token is not None:
                                try:
                                    await on_token(content)
                                except Exception:
                                    pass
        except (RateLimitError, HTTPError):
            raise
        except Exception as e:
            raise

        end = time.perf_counter()
        total_ms = round((end - start) * 1000, 2)
        first_token_ms = round((first_token - start) * 1000, 2) if first_token is not None else None
        return ModelResponse(
            text="".join(collected),
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            completion_tokens=(usage or {}).get("completion_tokens"),
            headers=headers,
            status=status,
        )

    async def chat(self, messages, temperature=0.0, max_tokens=512, model_override=None) -> ModelResponse:
        return await self.stream(messages, temperature=temperature, max_tokens=max_tokens, model_override=model_override)
