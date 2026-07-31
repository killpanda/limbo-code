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
async def test_normal_stream_terminated_by_blank_line():
    events = await _events(
        'data: {"type": "a"}\n\ndata: {"type": "b"}\n\n'
    )
    assert [e["type"] for e in events] == ["a", "b"]


@pytest.mark.asyncio
async def test_ignores_event_id_retry_and_comment_lines():
    # Only the payload's own fields drive dispatch; `event:`/`id:`/`retry:`
    # and comment (`: ping`) lines are dropped.
    text = (
        ": ping\n"
        'event: message\n'
        'data: {"type": "ok"}\n'
        "id: 42\n"
        "retry: 1000\n\n"
    )
    events = await _events(text)
    assert events == [{"type": "ok"}]


@pytest.mark.asyncio
async def test_multiline_data_payload_joined():
    # A single event split across several `data:` lines is joined with \n
    # before parsing (used for large JSON payloads).
    text = 'data: {"type": "x",\ndata: "v": 1}\n\n'
    events = await _events(text)
    assert events == [{"type": "x", "v": 1}]


@pytest.mark.asyncio
async def test_final_event_without_trailing_blank_line_is_not_lost():
    # The regression: the stream ends right after the last data line, with no
    # blank line to trigger the in-loop flush. The usage/finish_reason event
    # for Anthropic/OpenAI is sent last, so this is exactly what gets dropped.
    events = await _events('data: {"type": "done", "usage": {"tokens": 5}}\n')
    assert events == [{"type": "done", "usage": {"tokens": 5}}]


@pytest.mark.asyncio
async def test_final_multiline_payload_without_trailing_blank_line():
    events = await _events('data: {"type": "a",\ndata: "n": 1}\n')
    assert events == [{"type": "a", "n": 1}]


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing():
    assert await _events("") == []


@pytest.mark.asyncio
async def test_malformed_payload_is_skipped():
    # Bad JSON is dropped silently (matches the original `continue` behaviour);
    # the well-formed event on either side still yields.
    events = await _events(
        'data: {"type": "before"}\n\n'
        "data: not json\n\n"
        'data: {"type": "after"}\n\n'
    )
    assert [e["type"] for e in events] == ["before", "after"]


@pytest.mark.asyncio
async def test_trailing_malformed_payload_skipped():
    # A malformed trailing event (no blank line) is dropped, not raised.
    assert await _events("data: not json\n") == []
