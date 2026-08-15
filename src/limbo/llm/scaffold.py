"""Shared scaffolding for dialect clients.

Only verbatim-repeated plumbing lives here — credential resolution, HTTP
timeouts, the retry wrapper, thinking-effort mapping, and image encoding.
Everything dialect-specific (request bodies, stream parsing) stays in the
client modules. Deliberately helper functions, not a base class: the three
clients share plumbing, not behavior.
"""

from __future__ import annotations

import base64
import warnings
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx

from limbo.config import Config
from limbo.llm.catalog import (
    ModelSpec,
    missing_api_key_message,
    resolve_api_key,
)
from limbo.llm.client import RequestHook
from limbo.llm.retry import RetryPolicy, stream_with_retry
from limbo.models import Attachment, LLMEvent, Message


def require_api_key(spec: ModelSpec, config: Config) -> str:
    """The provider's API key, or a ValueError explaining where to set it."""
    api_key = resolve_api_key(spec, config)
    if not api_key:
        raise ValueError(missing_api_key_message(spec, config))
    return api_key


def http_timeout(config: Config) -> httpx.Timeout:
    """The configured request/connect timeout for LLM HTTP calls."""
    return httpx.Timeout(
        config.llm.timeout, connect=config.llm.connect_timeout
    )


def chat_with_retry(
    inner: Callable[..., AsyncIterator[LLMEvent]],
    config: Config,
    messages: list[Message],
    tools: list[dict[str, Any]],
    on_request: RequestHook | None,
) -> AsyncIterator[LLMEvent]:
    """Wrap one streaming attempt with the configured retry policy."""
    policy = RetryPolicy.from_config(config.llm)
    return stream_with_retry(lambda: inner(messages, tools, on_request), policy)


def map_thinking_effort(spec: ModelSpec, effort: str) -> str | None:
    """Map an effort level to the provider value.

    A level absent from the spec's ``thinking_levels`` is unsupported by
    the model: warn and return None so the caller omits the parameter.
    """
    value = spec.thinking_levels.get(effort)
    if value is None:
        warnings.warn(
            f"thinking_effort={effort!r} is not supported by {spec.id} "
            f"(supported: {sorted(spec.thinking_levels)}); ignoring.",
            stacklevel=2,
        )
    return value


def encode_image_data(attachment: Attachment) -> tuple[str, str] | None:
    """(base64 data, mime) for an image attachment; None when unreadable.

    Missing files (expired session attachments) degrade to None so a
    restored session can still be replayed; each dialect wraps the data in
    its own block shape.
    """
    try:
        data = base64.standard_b64encode(
            Path(attachment.path).read_bytes()
        ).decode("ascii")
    except OSError:
        return None
    return data, attachment.mime or "image/png"
