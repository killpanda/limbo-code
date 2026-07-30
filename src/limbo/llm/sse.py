"""Shared SSE parsing for the plain-httpx LLM clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


async def iter_sse(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed ``data:`` payloads from an SSE stream.

    ``event:``/``id:``/``retry:`` lines and comments (``: ping``) are
    ignored; the payload's own ``type`` field drives dispatch. Both the
    Anthropic Messages and OpenAI Responses streams follow this convention.
    """
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
