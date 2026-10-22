"""Unit tests for limbo.llm.retry (no network; sleeps are monkeypatched)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from openai import (
    APIStatusError,
    RateLimitError,
)

from limbo.llm.retry import (
    RetryPolicy,
    stream_with_retry,
)
from limbo.models import TextChunk


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://test/v1/chat/completions")


def _status_error(status: int, headers: dict[str, str] | None = None) -> APIStatusError:
    resp = httpx.Response(status, headers=headers or {}, request=_request())
    return APIStatusError(f"status {status}", response=resp, body=None)


def _rate_limit(headers: dict[str, str] | None = None) -> RateLimitError:
    resp = httpx.Response(429, headers=headers or {}, request=_request())
    return RateLimitError("rate limited", response=resp, body=None)


# ---------------------------------------------------------------------------
# is_retryable classification matrix
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# retry_after / parse_retry_after
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# compute_delay
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# stream_with_retry
# ---------------------------------------------------------------------------


def _make_factory(behaviors: list[tuple[list[TextChunk], BaseException | None]]):
    """Each entry: events to yield, then optional error to raise."""
    calls = 0

    def factory():
        nonlocal calls
        events, error = behaviors[min(calls, len(behaviors) - 1)]
        calls += 1

        async def gen():
            for e in events:
                yield e
            if error is not None:
                raise error

        return gen()

    return factory, lambda: calls


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures(sleeps: list[float]):
    events = [TextChunk(text="hello")]
    factory, calls = _make_factory(
        [([], _rate_limit()), ([], _status_error(503)), (events, None)]
    )
    result = [e async for e in stream_with_retry(factory, RetryPolicy())]
    assert result == events
    assert calls() == 3
    assert len(sleeps) == 2


@pytest.mark.asyncio
async def test_non_retryable_error_raises_immediately(sleeps: list[float]):
    factory, calls = _make_factory([([], _status_error(400))])
    with pytest.raises(APIStatusError):
        [e async for e in stream_with_retry(factory, RetryPolicy())]
    assert calls() == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# friendly_message
# ---------------------------------------------------------------------------


