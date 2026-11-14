"""Model endpoint classification tests: unreachable/404/405 => error, never retried."""

import asyncio
import time

import pytest

from relay.model_client import ModelClient, ModelEndpointUnreachable
from relay.scheduler import Scheduler
from relay.trace import Trace


def test_unreachable_connection_raises_endpoint_error():
    # 127.0.0.1:1 refuses connections
    client = ModelClient(base_url="http://127.0.0.1:1/openai/v1", api_key="x", model="m", timeout=2.0)

    async def run():
        await client.stream([{"role": "user", "content": "hi"}])

    with pytest.raises(ModelEndpointUnreachable) as ei:
        asyncio.run(run())
    assert "unreachable" in str(ei.value)
    asyncio.run(client.aclose())


def test_connection_refused_raises_endpoint_error_not_http_error():
    # same class of failure as a 404/405 on the completions path: classified
    # as ModelEndpointUnreachable, distinct from HTTPError
    client = ModelClient(base_url="http://127.0.0.1:1/openai/v1", api_key="x", model="m", timeout=2.0)

    async def run():
        await client.stream([{"role": "user", "content": "hi"}])

    with pytest.raises(ModelEndpointUnreachable) as ei:
        asyncio.run(run())
    from relay.model_client import HTTPError
    assert not isinstance(ei.value, HTTPError)
    asyncio.run(client.aclose())


class _StubClient:
    """Raises 405 like the static mount did."""

    def __init__(self):
        self.calls = 0

    async def stream(self, *a, **kw):
        self.calls += 1
        raise ModelEndpointUnreachable("model endpoint not served (HTTP 405)", status=405)


def test_scheduler_never_retries_unreachable():
    sched = Scheduler(rpm=30, tpm=6000, rpd=0, give_up_time=5.0, retry_budget=3)
    stub = _StubClient()
    step = Trace(question="q").new_step("decide")

    async def run():
        with pytest.raises(ModelEndpointUnreachable):
            await sched.call_model(stub, [{"role": "user", "content": "hi"}], step, deadline=time.time() + 5.0)

    asyncio.run(run())
    assert stub.calls == 1, f"expected exactly 1 attempt (no retry), got {stub.calls}"
    assert step.extra["scheduler_decision"]["reason"] == "model_endpoint_unreachable"
