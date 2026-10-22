"""Tests for the shared SSE parser (limbo.llm.sse).

Focuses on the final-flush fix: a stream closed without a trailing blank
line must still yield its last buffered event, since that event typically
carries ``usage`` / ``finish_reason`` for the Anthropic and Responses APIs.
"""

from __future__ import annotations

import httpx
import pytest

from limbo.llm.sse import iter_sse


async def _events(text: str) -> list[dict]:
    resp = httpx.Response(
        200,
        text=text,
        headers={"Content-Type": "text/event-stream"},
    )
    return [event async for event in iter_sse(resp)]


@pytest.mark.asyncio
async def test_final_event_without_trailing_blank_line_is_not_lost():
    # The regression: the stream ends right after the last data line, with no
    # blank line to trigger the in-loop flush. The usage/finish_reason event
    # for Anthropic/OpenAI is sent last, so this is exactly what gets dropped.
    events = await _events('data: {"type": "done", "usage": {"tokens": 5}}\n')
    assert events == [{"type": "done", "usage": {"tokens": 5}}]


