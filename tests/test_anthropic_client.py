"""Tests for the Anthropic Messages client (Kimi For Coding dialect)."""

import json

import pytest
import respx
from httpx import Response

from limbo.config import Config
from limbo.llm.anthropic_client import (
    AnthropicMessagesClient,
    _messages_to_anthropic,
    _tool_to_anthropic,
)
from limbo.models import CompletionMeta, Message, TextChunk, ThinkingChunk, ToolCallEvent

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


@pytest.mark.asyncio
@respx.mock
async def test_request_headers_and_body(client):
    client.config.llm.thinking_effort = "high"
    route = respx.post(MESSAGES_URL).mock(
        return_value=_sse(
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "ok"}},
            {"type": "message_stop"},
        )
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    await _collect(
        client.chat(
            [Message(role="system", content="sys"), Message(role="user", content="hi")],
            tools=tools,
        )
    )

    request = route.calls.last.request
    assert request.headers["x-api-key"] == "sk-kim-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["user-agent"] == "KimiCLI/1.5"

    body = json.loads(request.content)
    assert body["model"] == "k3"
    assert body["max_tokens"] == 131_072
    assert body["system"] == "sys"
    assert body["stream"] is True
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    # Temperature is incompatible with thinking.
    assert "temperature" not in body
    assert body["tools"] == [
        {
            "name": "read",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


@pytest.mark.asyncio
@respx.mock
async def test_thinking_effort_defaults_to_provider_behavior(client):
    route = respx.post(MESSAGES_URL).mock(return_value=_sse({"type": "message_stop"}))
    await _collect(client.chat([Message(role="user", content="hi")], tools=[]))
    body = json.loads(route.calls.last.request.content)
    assert body["thinking"]["type"] == "adaptive"
    assert "output_config" not in body


@pytest.mark.asyncio
@respx.mock
async def test_unsupported_effort_warns_and_omits(client):
    client.config.llm.thinking_effort = "off"
    route = respx.post(MESSAGES_URL).mock(return_value=_sse({"type": "message_stop"}))
    with pytest.warns(UserWarning, match="not supported"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))
    body = json.loads(route.calls.last.request.content)
    assert "output_config" not in body


@pytest.mark.asyncio
@respx.mock
async def test_http_error_raises_with_body(client):
    respx.post(MESSAGES_URL).mock(
        return_value=Response(
            401,
            json={"error": {"message": "Invalid Authentication",
                            "type": "invalid_authentication_error"}},
        )
    )
    with pytest.raises(RuntimeError, match="401.*Invalid Authentication"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))


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


def test_tool_definition_conversion():
    tool = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {"type": "object"},
        },
    }
    assert _tool_to_anthropic(tool) == {
        "name": "bash",
        "description": "Run a command",
        "input_schema": {"type": "object"},
    }


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


def test_thinking_replay_falls_back_to_empty_signature():
    message = Message(role="assistant", content="hi", reasoning="thoughts")
    _, converted = _messages_to_anthropic([message])
    assert converted[0]["content"][0] == {
        "type": "thinking",
        "thinking": "thoughts",
        "signature": "",
    }


def test_consecutive_tool_results_merge_into_one_user_message():
    messages = [
        Message(role="assistant", content=None, tool_calls=[
            {"id": "toolu_1", "type": "function",
             "function": {"name": "read", "arguments": {}}},
            {"id": "toolu_2", "type": "function",
             "function": {"name": "ls", "arguments": {}}},
        ]),
        Message(role="tool", content="file contents", tool_call_id="toolu_1"),
        Message(role="tool", content="dir listing", tool_call_id="toolu_2"),
    ]
    _, converted = _messages_to_anthropic(messages)
    assert converted[1] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": "file contents"},
            {"type": "tool_result", "tool_use_id": "toolu_2",
             "content": "dir listing"},
        ],
    }


def test_system_messages_join_into_system_param():
    messages = [
        Message(role="system", content="part 1"),
        Message(role="system", content="part 2"),
        Message(role="user", content="hi"),
    ]
    system, converted = _messages_to_anthropic(messages)
    assert system == "part 1\n\npart 2"
    assert converted == [{"role": "user", "content": "hi"}]


# -- response metadata -------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_completion_meta_captures_usage_and_stop_reason(client):
    respx.post(MESSAGES_URL).mock(
        return_value=_sse(
            {
                "type": "message_start",
                "message": {
                    "usage": {"input_tokens": 50, "cache_read_input_tokens": 30}
                },
            },
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "ok"}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 7},
            },
            {"type": "message_stop"},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    meta = next(e for e in events if isinstance(e, CompletionMeta))
    assert meta.usage == {
        "input_tokens": 50,
        "cache_read_input_tokens": 30,
        "output_tokens": 7,
    }
    assert meta.finish_reason == "end_turn"
    assert meta.ttft is not None
    assert meta.duration is not None


@pytest.mark.asyncio
@respx.mock
async def test_on_request_hook_receives_exact_body(client):
    respx.post(MESSAGES_URL).mock(return_value=_sse({"type": "message_stop"}))
    bodies = []

    await _collect(
        client.chat(
            [Message(role="system", content="sys"), Message(role="user", content="hi")],
            tools=[],
            on_request=bodies.append,
        )
    )

    assert len(bodies) == 1
    body = bodies[0]
    assert body["model"] == "k3"
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["stream"] is True
