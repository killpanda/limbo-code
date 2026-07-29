"""Unified retry helper for LLM streaming requests.

Shared by both provider clients (OpenAI-compatible and Anthropic) so that
429/5xx/connection/timeout handling behaves identically on every path.

Core contract of :func:`stream_with_retry`: retries only happen *before the
first event is yielded*. Once any event has been emitted to the consumer,
subsequent exceptions propagate untouched — this makes duplicate streamed
output (and thus duplicated UI text) impossible by construction.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

from limbo.config import LLMConfig
from limbo.models import LLMEvent

# Hard cap for a single backoff sleep. Not configurable in the MVP: it bounds
# the total wait budget of one turn (~90s worst case with max_retries=3).
DEFAULT_MAX_DELAY = 30.0


@dataclass(frozen=True)
class RetryPolicy:
    """Retry tuning: max_retries=3 means at most 4 attempts total."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = DEFAULT_MAX_DELAY

    @classmethod
    def from_config(cls, cfg: LLMConfig) -> RetryPolicy:
        return cls(max_retries=cfg.max_retries, base_delay=cfg.retry_base_delay)


class LLMHttpError(Exception):
    """Normalized non-200 HTTP response, carrying structured retry metadata."""

    def __init__(
        self,
        status_code: int,
        message: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


class LLMOverloadedError(Exception):
    """Provider reports overload (e.g. Anthropic SSE ``overloaded_error``)."""


def is_retryable(exc: BaseException) -> bool:
    """Classify whether retrying the request could succeed.

    Retryable: 429, 408, 5xx, connection errors, read/connect timeouts,
    provider overload. Everything else (other 4xx, cancellations) is not.
    """
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        return False
    if isinstance(exc, LLMOverloadedError):
        return True
    if isinstance(exc, LLMHttpError):
        return _status_retryable(exc.status_code)
    if isinstance(exc, APIStatusError):
        # Includes RateLimitError (429) and InternalServerError (5xx).
        return _status_retryable(exc.status_code)
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout)):
        return True
    return False


def _status_retryable(status_code: int) -> bool:
    return status_code in (408, 429) or status_code >= 500


def retry_after(exc: BaseException) -> float | None:
    """Extract the server-provided Retry-After (seconds), if any."""
    if isinstance(exc, LLMHttpError):
        return exc.retry_after
    if isinstance(exc, APIStatusError):
        return parse_retry_after(exc.response.headers.get("retry-after"))
    return None


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value (delta-seconds or HTTP-date).

    Returns None on any parse failure — a malformed header must never
    break the retry flow.
    """
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        date = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if date is None:
        return None
    if date.tzinfo is None:
        # HTTP-date is always GMT; assume UTC if parsing dropped the tz.
        date = date.replace(tzinfo=timezone.utc)
    return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())


def compute_delay(
    attempt: int,
    policy: RetryPolicy,
    retry_after: float | None = None,
) -> float:
    """Full-jitter exponential backoff: uniform(0, min(max, base * 2**attempt)).

    A server-provided Retry-After raises the floor (max of the two), still
    capped at max_delay — bounding the total wait budget of a turn.
    """
    computed = random.uniform(0.0, min(policy.max_delay, policy.base_delay * 2**attempt))
    if retry_after is not None:
        computed = max(retry_after, computed)
    return min(policy.max_delay, computed)


async def stream_with_retry(
    factory: Callable[[], AsyncIterator[LLMEvent]],
    policy: RetryPolicy,
) -> AsyncIterator[LLMEvent]:
    """Yield events from factory(), retrying pre-first-event failures.

    Each attempt calls factory() to build a fresh async generator (a brand
    new HTTP request). Exceptions raised before the first yield are retried
    when :func:`is_retryable`; once anything has been yielded, exceptions
    propagate untouched. CancelledError/KeyboardInterrupt always propagate
    immediately (they are BaseExceptions, never caught here).
    """
    attempt = 0
    while True:
        yielded = False
        try:
            async for event in factory():
                yielded = True
                yield event
            return
        except Exception as e:
            if yielded or attempt >= policy.max_retries or not is_retryable(e):
                raise
            await asyncio.sleep(compute_delay(attempt, policy, retry_after(e)))
            attempt += 1


def friendly_message(exc: BaseException) -> str | None:
    """User-facing Chinese hint for an exhausted/unretryable LLM failure.

    Returns None when there is no better advice than the raw error text;
    the raw exception always stays in the trace log either way.
    """
    if isinstance(exc, LLMOverloadedError):
        return "模型服务过载，已自动重试仍失败，可稍后重发上一条消息"
    if isinstance(exc, LLMHttpError):
        if exc.status_code == 429:
            return "服务限流，已自动重试仍失败，可稍后重发上一条消息"
        if exc.status_code >= 500:
            return "模型服务异常，已自动重试仍失败，可稍后重发上一条消息"
        if exc.status_code == 408:
            return "请求超时，已自动重试仍失败，请检查网络后重发上一条消息"
        return None
    if isinstance(exc, APIStatusError):
        # RateLimitError (429) is an APIStatusError subclass.
        if exc.status_code == 429:
            return "服务限流，已自动重试仍失败，可稍后重发上一条消息"
        if exc.status_code >= 500:
            return "模型服务异常，已自动重试仍失败，可稍后重发上一条消息"
        if exc.status_code == 408:
            return "请求超时，已自动重试仍失败，请检查网络后重发上一条消息"
        return None
    if isinstance(exc, APITimeoutError) or isinstance(
        exc, (httpx.ReadTimeout, httpx.ConnectTimeout)
    ):
        return "请求超时，已自动重试仍失败，请检查网络后重发上一条消息"
    if isinstance(exc, (APIConnectionError, httpx.ConnectError)):
        return "网络连接异常，已自动重试仍失败，请检查网络后重发上一条消息"
    return None
