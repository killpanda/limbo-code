"""Shared SSE parsing for the plain-httpx LLM clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx


async def iter_sse(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed ``data:`` payloads from an SSE stream.

    ``event:``/``id:``/``retry:`` lines and comments (``: ping``) are
    ignored; the payload's own ``type`` field drives dispatch. Both the
    Anthropic Messages and OpenAI Responses streams follow this convention.

    A final flush after the stream ends catches a trailing event that the
    server closed without a terminating blank line — that last event often
    carries ``usage`` / ``finish_reason``, so dropping it would desync the
    compaction trigger and token counters.
    """
    data_lines: list[str] = []

    def _flush() -> dict[str, Any] | None:
        if not data_lines:
            return None
        payload = "\n".join(data_lines)
        data_lines.clear()
        try:
            return cast(dict[str, Any], json.loads(payload))
        except json.JSONDecodeError:
            return None

    async for line in resp.aiter_lines():
        if not line:
            event = _flush()
            if event is not None:
                yield event
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
    # The server may close the connection without a trailing blank line; a
    # buffered event would otherwise be lost. Anthropic/OpenAI both put
    # usage + finish_reason in this final event.
    event = _flush()
    if event is not None:
        yield event
