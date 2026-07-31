"""Tests for the Anthropic Messages client (Kimi For Coding dialect)."""

import asyncio
import base64
import json

import httpx
import pytest
import respx
from httpx import Response

from limbo.config import Config
from limbo.llm.anthropic_client import (
    AnthropicMessagesClient,
    _messages_to_anthropic,
    _tool_to_anthropic,
)
from limbo.llm.retry import LLMHttpError, LLMOverloadedError
from limbo.models import (
    Attachment,
    CompletionMeta,
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
    with pytest.raises(LLMHttpError, match="401.*Invalid Authentication"):
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


def test_thinking_replay_skips_foreign_dialect_signature():
    # A Responses-dialect signature (stored before a mid-session model
    # switch) must not be replayed as an Anthropic signature — that 400s.
    message = Message(
        role="assistant",
        content="hi",
        reasoning="thoughts",
        reasoning_signature='responses:{"type": "reasoning", "id": "rs_1"}',
    )
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


# -- unified retry integration (limbo.llm.retry) -----------------------------


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace asyncio.sleep with a recorder so retries cost no real time."""
    calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return calls


def _text_ok() -> Response:
    return _sse(
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_stop"},
    )


def test_httpx_timeout_wired_from_config(client):
    client.config.llm.timeout = 123.0
    client.config.llm.connect_timeout = 4.0

    timeout = client.client.timeout
    assert timeout.read == 123.0
    assert timeout.connect == 4.0


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_429_then_succeeds(client, sleeps):
    route = respx.post(MESSAGES_URL).mock(
        side_effect=[
            Response(
                429,
                json={"error": {"type": "rate_limit_error",
                                "message": "slow down"}},
                headers={"retry-after": "0"},
            ),
            _text_ok(),
        ]
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 2
    assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "ok"
    assert len(sleeps) == 1


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_500_then_succeeds(client, sleeps):
    route = respx.post(MESSAGES_URL).mock(
        side_effect=[
            Response(500, json={"error": {"type": "api_error",
                                        "message": "internal"}}),
            _text_ok(),
        ]
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 2
    assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "ok"


@pytest.mark.asyncio
@respx.mock
async def test_400_is_not_retried(client, sleeps):
    route = respx.post(MESSAGES_URL).mock(
        return_value=Response(
            400,
            json={"error": {"type": "invalid_request_error",
                            "message": "bad request"}},
        )
    )

    with pytest.raises(LLMHttpError, match="400.*bad request"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 1
    assert sleeps == []


@pytest.mark.asyncio
@respx.mock
async def test_http_error_carries_retry_after(client):
    client.config.llm.max_retries = 0
    respx.post(MESSAGES_URL).mock(
        return_value=Response(
            429,
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "7"},
        )
    )

    with pytest.raises(LLMHttpError) as excinfo:
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 7.0


@pytest.mark.asyncio
@respx.mock
async def test_sse_overloaded_error_is_retried(client, sleeps):
    route = respx.post(MESSAGES_URL).mock(
        side_effect=[
            _sse({"type": "error",
                  "error": {"type": "overloaded_error", "message": "Overloaded"}}),
            _text_ok(),
        ]
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 2
    assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "ok"


@pytest.mark.asyncio
@respx.mock
async def test_sse_overloaded_exhausts_retries_then_raises(client, sleeps):
    respx.post(MESSAGES_URL).mock(
        return_value=_sse(
            {"type": "error",
             "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
    )

    with pytest.raises(LLMOverloadedError, match="overloaded_error"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))


@pytest.mark.asyncio
@respx.mock
async def test_sse_invalid_request_error_is_not_retried(client, sleeps):
    route = respx.post(MESSAGES_URL).mock(
        return_value=_sse(
            {"type": "error",
             "error": {"type": "invalid_request_error", "message": "bad input"}}
        )
    )

    with pytest.raises(RuntimeError, match="invalid_request_error"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 1
    assert sleeps == []


class _DroppingStream(httpx.AsyncByteStream):
    """Yields its chunks, then drops the connection mid-stream."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk
        raise httpx.ReadError("connection dropped")

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
@respx.mock
async def test_mid_stream_drop_after_first_event_is_not_retried(client, sleeps):
    first_event = (
        b"event: content_block_delta\n"
        b'data: {"type": "content_block_delta", "index": 0, '
        b'"delta": {"type": "text_delta", "text": "partial"}}\n\n'
    )
    route = respx.post(MESSAGES_URL).mock(
        side_effect=[
            Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                stream=_DroppingStream([first_event]),
            ),
            _text_ok(),
        ]
    )

    with pytest.raises(httpx.ReadError):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    # The first text delta already reached the consumer: retrying would
    # duplicate output, so the exception passes through with no 2nd request.
    assert route.call_count == 1
    assert sleeps == []


def test_user_message_with_image_attachment_becomes_blocks(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png-bytes")
    message = Message(
        role="user",
        content="看这张图 [图片 #1]",
        attachments=[
            Attachment(
                kind="image", name="shot.png", path=str(image), mime="image/png"
            )
        ],
    )
    _, converted = _messages_to_anthropic([message])
    content = converted[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(b"png-bytes").decode("ascii"),
        },
    }
    assert content[1] == {"type": "text", "text": "看这张图 [图片 #1]"}


def test_user_message_missing_image_falls_back_to_plain_text(tmp_path):
    message = Message(
        role="user",
        content="看这张图 [图片 #1]",
        attachments=[
            Attachment(
                kind="image",
                name="gone.png",
                path=str(tmp_path / "gone.png"),
                mime="image/png",
            )
        ],
    )
    _, converted = _messages_to_anthropic([message])
    assert converted[0]["content"] == "看这张图 [图片 #1]"


def test_user_message_without_attachments_stays_plain_string():
    _, converted = _messages_to_anthropic([Message(role="user", content="hi")])
    assert converted[0]["content"] == "hi"


def test_tool_message_with_image_attachment_becomes_tool_result_blocks(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png-bytes")
    message = Message(
        role="tool",
        content="Read image file [image/png]",
        tool_call_id="c1",
        attachments=[
            Attachment(
                kind="image", name="shot.png", path=str(image), mime="image/png"
            )
        ],
    )
    _, converted = _messages_to_anthropic([message])
    block = converted[0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "c1"
    content = block["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1] == {"type": "text", "text": "Read image file [image/png]"}


def test_tool_message_missing_image_stays_plain_string(tmp_path):
    message = Message(
        role="tool",
        content="Read image file [image/png]",
        tool_call_id="c1",
        attachments=[
            Attachment(kind="image", name="gone.png", path=str(tmp_path / "gone.png"))
        ],
    )
    _, converted = _messages_to_anthropic([message])
    block = converted[0]["content"][0]
    assert block["content"] == "Read image file [image/png]"
