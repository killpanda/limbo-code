"""Tests for the Anthropic Messages client (Kimi For Coding dialect)."""

import asyncio
import json

import pytest
import respx
from httpx import Response

from limbo.config import Config
from limbo.llm.anthropic_client import (
    AnthropicMessagesClient,
    _messages_to_anthropic,
)
from limbo.models import (
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
)

MESSAGES_URL = "https://api.kimi.com/coding/v1/messages"


@pytest.fixture
def client():
    cfg = Config()
    cfg.llm.api_key = "sk-kim-test"
    cfg.llm.model = "k3"
    return AnthropicMessagesClient(cfg)


async def _collect(chat):
    return [event async for event in chat]


def _sse(*events: dict) -> Response:
    payload = "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    )
    return Response(
        200, text=payload, headers={"Content-Type": "text/event-stream"}
    )


# -- request shape -----------------------------------------------------------


# -- streaming ---------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_streams_thinking_text_and_tool_call(client):
    respx.post(MESSAGES_URL).mock(
        return_value=_sse(
            {"type": "message_start", "message": {"id": "msg_1"}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "let me "}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "thinking_delta", "thinking": "think"}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "signature_delta", "signature": "sig_abc"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "text_delta", "text": "answer"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_start", "index": 2,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "read"}},
            {"type": "content_block_delta", "index": 2,
             "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}},
            {"type": "content_block_delta", "index": 2,
             "delta": {"type": "input_json_delta", "partial_json": " \"a.py\"}"}},
            {"type": "content_block_stop", "index": 2},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    thinking = "".join(e.text for e in events if isinstance(e, ThinkingChunk))
    signatures = [e.signature for e in events if isinstance(e, ThinkingChunk) and e.signature]
    text = "".join(e.text for e in events if isinstance(e, TextChunk))
    calls = [e for e in events if isinstance(e, ToolCallEvent)]

    assert thinking == "let me think"
    assert signatures == ["sig_abc"]
    assert text == "answer"
    assert len(calls) == 1
    assert calls[0].id == "toolu_1"
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "a.py"}


# -- message conversion ------------------------------------------------------


def test_assistant_replays_thinking_then_text_then_tool_use():
    message = Message(
        role="assistant",
        content="done",
        reasoning="my thoughts",
        reasoning_signature="sig_1",
        tool_calls=[
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "read", "arguments": {"path": "a.py"}},
            }
        ],
    )
    system, converted = _messages_to_anthropic([message])
    assert system == ""
    assert converted == [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "my thoughts", "signature": "sig_1"},
                {"type": "text", "text": "done"},
                {"type": "tool_use", "id": "toolu_1", "name": "read",
                 "input": {"path": "a.py"}},
            ],
        }
    ]


# -- response metadata -------------------------------------------------------


# -- unified retry integration (limbo.llm.retry) -----------------------------


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace asyncio.sleep with a recorder so retries cost no real time."""
    calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return calls


