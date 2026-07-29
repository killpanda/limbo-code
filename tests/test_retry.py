"""Unit tests for limbo.llm.retry (no network; sleeps are monkeypatched)."""

from __future__ import annotations

import asyncio
import random
import time
from email.utils import formatdate

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from limbo.config import LLMConfig
from limbo.llm.retry import (
    LLMHttpError,
    LLMOverloadedError,
    RetryPolicy,
    compute_delay,
    friendly_message,
    is_retryable,
    parse_retry_after,
    retry_after,
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


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_rate_limit(), True),
        (_status_error(408), True),
        (_status_error(500), True),
        (_status_error(503), True),
        (_status_error(400), False),
        (_status_error(401), False),
        (_status_error(404), False),
        (_status_error(409), False),
        (APIConnectionError(message="conn", request=_request()), True),
        (APITimeoutError(request=_request()), True),
        (LLMHttpError(429), True),
        (LLMHttpError(408), True),
        (LLMHttpError(500), True),
        (LLMHttpError(400), False),
        (LLMHttpError(401), False),
        (LLMOverloadedError("overloaded"), True),
        (httpx.ConnectError("boom", request=_request()), True),
        (httpx.ReadTimeout("boom", request=_request()), True),
        (httpx.ConnectTimeout("boom", request=_request()), True),
        (asyncio.CancelledError(), False),
        (KeyboardInterrupt(), False),
        (ValueError("nope"), False),
        (RuntimeError("nope"), False),
    ],
)
def test_is_retryable(exc: BaseException, expected: bool):
    assert is_retryable(exc) is expected


# ---------------------------------------------------------------------------
# retry_after / parse_retry_after
# ---------------------------------------------------------------------------


def test_retry_after_from_normalized_error():
    assert retry_after(LLMHttpError(429, retry_after=5.0)) == 5.0
    assert retry_after(LLMHttpError(429)) is None


def test_retry_after_from_openai_error_delta_seconds():
    assert retry_after(_rate_limit({"retry-after": "5"})) == 5.0


def test_retry_after_from_openai_error_http_date():
    headers = {"retry-after": formatdate(time.time() + 5, usegmt=True)}
    value = retry_after(_rate_limit(headers))
    assert value is not None
    assert 3.0 < value <= 5.0


def test_retry_after_garbage_and_missing_headers():
    assert retry_after(_rate_limit({"retry-after": "not-a-date"})) is None
    assert retry_after(_rate_limit()) is None
    assert retry_after(ValueError("x")) is None


def test_parse_retry_after_past_http_date_clamps_to_zero():
    headers = formatdate(time.time() - 60, usegmt=True)
    assert parse_retry_after(headers) == 0.0


# ---------------------------------------------------------------------------
# compute_delay
# ---------------------------------------------------------------------------


def _capture_uniform(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []

    def fake_uniform(a: float, b: float) -> float:
        bounds.append((a, b))
        return 0.0

    monkeypatch.setattr(random, "uniform", fake_uniform)
    return bounds


def test_compute_delay_exponential_bounds(monkeypatch: pytest.MonkeyPatch):
    bounds = _capture_uniform(monkeypatch)
    policy = RetryPolicy(base_delay=1.0)
    for attempt, expected_upper in ((0, 1.0), (1, 2.0), (2, 4.0)):
        compute_delay(attempt, policy)
        assert bounds[-1] == (0.0, expected_upper)


def test_compute_delay_capped_at_max_delay(monkeypatch: pytest.MonkeyPatch):
    bounds = _capture_uniform(monkeypatch)
    policy = RetryPolicy(base_delay=1.0, max_delay=30.0)
    compute_delay(10, policy)  # 2**10 = 1024 >> 30
    assert bounds[-1] == (0.0, 30.0)


def test_compute_delay_retry_after_raises_floor(monkeypatch: pytest.MonkeyPatch):
    _capture_uniform(monkeypatch)  # uniform returns 0.0
    policy = RetryPolicy(base_delay=1.0)
    assert compute_delay(0, policy, retry_after=5.0) == 5.0


def test_compute_delay_retry_after_capped_at_max_delay(monkeypatch: pytest.MonkeyPatch):
    _capture_uniform(monkeypatch)
    policy = RetryPolicy(max_delay=30.0)
    assert compute_delay(0, policy, retry_after=120.0) == 30.0


def test_retry_policy_from_config():
    cfg = LLMConfig(max_retries=5, retry_base_delay=2.0)
    policy = RetryPolicy.from_config(cfg)
    assert policy.max_retries == 5
    assert policy.base_delay == 2.0
    assert policy.max_delay == 30.0


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


@pytest.mark.asyncio
async def test_error_after_first_event_passes_through(sleeps: list[float]):
    """Core regression: once an event was yielded, no retry may happen."""
    factory, calls = _make_factory([([TextChunk(text="partial")], _rate_limit())])
    collected = []
    with pytest.raises(RateLimitError):
        async for e in stream_with_retry(factory, RetryPolicy()):
            collected.append(e)
    assert collected == [TextChunk(text="partial")]
    assert calls() == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_max_retries_zero_disables_retry(sleeps: list[float]):
    factory, calls = _make_factory([([], _rate_limit())])
    with pytest.raises(RateLimitError):
        [e async for e in stream_with_retry(factory, RetryPolicy(max_retries=0))]
    assert calls() == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_exhausted_retries_raise_last_error(sleeps: list[float]):
    factory, calls = _make_factory([([], _status_error(500))])
    policy = RetryPolicy(max_retries=3)
    with pytest.raises(APIStatusError):
        [e async for e in stream_with_retry(factory, policy)]
    assert calls() == 4  # max_retries + 1 attempts
    assert len(sleeps) == 3


@pytest.mark.asyncio
async def test_cancelled_error_passes_through(sleeps: list[float]):
    """Cancellation must never be retried or delayed."""
    factory, calls = _make_factory([([], asyncio.CancelledError())])
    with pytest.raises(asyncio.CancelledError):
        [e async for e in stream_with_retry(factory, RetryPolicy())]
    assert calls() == 1
    assert sleeps == []


# ---------------------------------------------------------------------------
# friendly_message
# ---------------------------------------------------------------------------


def test_friendly_message_rate_limit():
    msg = friendly_message(_rate_limit())
    assert msg is not None
    assert "限流" in msg and "稍后重发" in msg
    msg = friendly_message(LLMHttpError(429))
    assert msg is not None and "限流" in msg


def test_friendly_message_server_error():
    assert "服务异常" in (friendly_message(_status_error(500)) or "")
    assert "服务异常" in (friendly_message(LLMHttpError(503)) or "")


def test_friendly_message_timeout_and_connection():
    assert "超时" in (friendly_message(APITimeoutError(request=_request())) or "")
    assert "超时" in (friendly_message(LLMHttpError(408)) or "")
    conn = APIConnectionError(message="c", request=_request())
    assert "网络连接异常" in (friendly_message(conn) or "")
    assert "网络连接异常" in (
        friendly_message(httpx.ConnectError("c", request=_request())) or ""
    )


def test_friendly_message_overloaded():
    assert "过载" in (friendly_message(LLMOverloadedError("x")) or "")


def test_friendly_message_returns_none_for_other_errors():
    assert friendly_message(_status_error(400)) is None
    assert friendly_message(LLMHttpError(401)) is None
    assert friendly_message(ValueError("x")) is None
