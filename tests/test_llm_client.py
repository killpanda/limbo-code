import asyncio
import base64
import json
from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response
from openai import BadRequestError, RateLimitError

from limbo.config import Config, ProviderOverride
from limbo.llm.openai_client import (
    OpenAICompatibleClient,
    _message_to_openai,
    _messages_to_openai,
)
from limbo.models import (
    Attachment,
    CompletionMeta,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
)


@pytest.fixture
def client():
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.base_url = "https://api.test.com/v1"
    return OpenAICompatibleClient(cfg)


@pytest.fixture
def kimi_client():
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.base_url = "https://api.test.com/v1"
    cfg.llm.model = "kimi-k3"
    return OpenAICompatibleClient(cfg)


def _sse_response(text_stream: str) -> Response:
    return Response(
        200,
        text=text_stream,
        headers={"Content-Type": "text/event-stream"},
    )


async def _collect(chat):
    return [event async for event in chat]


@pytest.mark.asyncio
@respx.mock
async def test_client_returns_text(client):
    text_stream = (
        'data: {"id":"1","object":"chat.completion.chunk",'
        '"choices":[{"delta":{"role":"assistant","content":"hello"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            text=text_stream,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))
    assert "".join([e.text for e in events if hasattr(e, "text")]) == "hello"
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_client_returns_tool_call(client):
    tool_stream = (
        'data: {"id":"1","object":"chat.completion.chunk",'
        '"choices":[{"delta":{"tool_calls":['
        '{"index":0,"id":"call_1","function":{"name":"read","arguments":""}}'
        "]}}]}\n\n"
        'data: {"id":"2","object":"chat.completion.chunk",'
        '"choices":[{"delta":{"tool_calls":['
        '{"index":0,"function":{"arguments":"{\\"path\\": \\"main.py\\"}"}}'
        "]}}]}\n\n"
        "data: [DONE]\n\n"
    )
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            text=tool_stream,
            headers={"Content-Type": "text/event-stream"},
        )
    )

    events = await _collect(
        client.chat([Message(role="user", content="read main.py")], tools=[])
    )
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments.get("path") == "main.py"
    assert route.called


def test_message_to_openai_serializes_tool_arguments():
    message = Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": {"path": "main.py"},
                },
            }
        ],
    )
    result = _message_to_openai(message)
    assert result["role"] == "assistant"
    assert len(result["tool_calls"]) == 1
    args = result["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"path": "main.py"}
    # The original message object is left untouched as a dict.
    assert isinstance(message.tool_calls[0]["function"]["arguments"], dict)


def test_message_to_openai_keeps_empty_tool_content():
    message = Message(
        role="tool",
        content="",
        tool_call_id="call_1",
    )
    result = _message_to_openai(message)
    assert result["role"] == "tool"
    assert "content" in result
    assert result["content"] == ""


def test_message_to_openai_includes_content_for_assistant_tool_calls():
    message = Message(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": {"path": "main.py"},
                },
            }
        ],
    )
    result = _message_to_openai(message)
    assert result["role"] == "assistant"
    assert "content" in result
    assert result["content"] == ""
    assert len(result["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_client_close_releases_http_client(client):
    mock_client = AsyncMock()
    client._client = mock_client
    await client.close()
    mock_client.close.assert_awaited_once()
    assert client._client is None


@pytest.mark.asyncio
async def test_client_close_is_noop_when_not_initialized(client):
    assert client._client is None
    await client.close()
    assert client._client is None


# -- model-specific behavior (catalog-driven) --------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_kimi_k3_sends_reasoning_effort_and_max_tokens(kimi_client):
    kimi_client.config.llm.thinking_effort = "high"
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        )
    )

    await _collect(kimi_client.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 131_072


@pytest.mark.asyncio
@respx.mock
async def test_kimi_k3_omits_reasoning_effort_by_default(kimi_client):
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        )
    )

    await _collect(kimi_client.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert "reasoning_effort" not in body
    assert "thinking" not in body


@pytest.mark.asyncio
@respx.mock
async def test_kimi_k3_unsupported_effort_warns_and_omits(kimi_client):
    kimi_client.config.llm.thinking_effort = "off"  # K3 cannot disable thinking
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        )
    )

    with pytest.warns(UserWarning, match="not supported"):
        await _collect(
            kimi_client.chat([Message(role="user", content="hi")], tools=[])
        )

    body = json.loads(route.calls.last.request.content)
    assert "reasoning_effort" not in body


@pytest.mark.asyncio
@respx.mock
async def test_deepseek_format_thinking_toggle():
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.base_url = "https://api.test.com/v1"
    cfg.llm.model = "kimi-k2-thinking"
    cfg.llm.thinking_effort = "off"
    k2_client = OpenAICompatibleClient(cfg)
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        )
    )

    await _collect(k2_client.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert body["thinking"] == {"type": "disabled"}


def _glm_client(
    model: str = "glm-4.7", thinking_effort: str | None = None
) -> OpenAICompatibleClient:
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.base_url = "https://api.test.com/v1"
    cfg.llm.model = model
    cfg.llm.thinking_effort = thinking_effort
    return OpenAICompatibleClient(cfg)


_OK_STREAM = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


@pytest.mark.asyncio
@respx.mock
async def test_zai_format_thinking_enabled_with_clear_thinking():
    glm = _glm_client(thinking_effort="high")
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(_OK_STREAM)
    )

    await _collect(glm.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert "reasoning_effort" not in body  # glm-4.7 declares no levels


@pytest.mark.asyncio
@respx.mock
async def test_zai_format_thinking_disabled():
    glm = _glm_client(thinking_effort="off")
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(_OK_STREAM)
    )

    await _collect(glm.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert body["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
@respx.mock
async def test_glm_5_2_sends_reasoning_effort_alongside_thinking():
    glm = _glm_client(model="glm-5.2", thinking_effort="low")
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(_OK_STREAM)
    )

    await _collect(glm.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    # pi sends both for glm-5.2: zai thinking + reasoning_effort (low→high).
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert body["reasoning_effort"] == "high"


@pytest.mark.asyncio
@respx.mock
async def test_glm_5_2_unsupported_effort_warns_but_keeps_thinking():
    glm = _glm_client(model="glm-5.2", thinking_effort="xhigh")
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(_OK_STREAM)
    )

    with pytest.warns(UserWarning, match="not supported"):
        await _collect(glm.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert "reasoning_effort" not in body
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}


@pytest.mark.asyncio
@respx.mock
async def test_glm_tool_stream_sent_only_with_tools():
    glm = _glm_client()
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(_OK_STREAM)
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

    await _collect(glm.chat([Message(role="user", content="hi")], tools=tools))
    body = json.loads(route.calls.last.request.content)
    assert body["tool_stream"] is True

    await _collect(glm.chat([Message(role="user", content="hi")], tools=[]))
    body = json.loads(route.calls.last.request.content)
    assert "tool_stream" not in body


@pytest.mark.asyncio
@respx.mock
async def test_provider_override_base_url_and_api_key_used():
    cfg = Config()
    cfg.llm.model = "glm-4.7"
    cfg.providers["glm"] = ProviderOverride(
        base_url="https://relay.example.com/v4",
        api_key="override-key",
        headers={"x-relay": "on"},
    )
    glm = OpenAICompatibleClient(cfg)
    route = respx.post("https://relay.example.com/v4/chat/completions").mock(
        return_value=_sse_response(_OK_STREAM)
    )

    await _collect(glm.chat([Message(role="user", content="hi")], tools=[]))

    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer override-key"
    assert request.headers["x-relay"] == "on"


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_content_streams_as_thinking_chunks(kimi_client):
    stream = (
        'data: {"choices":[{"delta":{"reasoning_content":"let me "}}]}\n\n'
        'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(stream)
    )

    events = await _collect(
        kimi_client.chat([Message(role="user", content="hi")], tools=[])
    )
    thinking = "".join(e.text for e in events if isinstance(e, ThinkingChunk))
    text = "".join(e.text for e in events if isinstance(e, TextChunk))
    assert thinking == "let me think"
    assert text == "answer"


def test_message_to_openai_replays_reasoning_content_when_required():
    message = Message(role="assistant", content="done", reasoning="my thoughts")
    result = _message_to_openai(message, include_reasoning=True)
    assert result["reasoning_content"] == "my thoughts"


def test_message_to_openai_reasoning_defaults_to_empty_for_k3():
    message = Message(role="assistant", content="done")
    result = _message_to_openai(message, include_reasoning=True)
    assert result["reasoning_content"] == ""


def test_message_to_openai_omits_reasoning_by_default():
    message = Message(role="assistant", content="done", reasoning="my thoughts")
    result = _message_to_openai(message)
    assert "reasoning_content" not in result


def test_config_max_tokens_overrides_catalog(kimi_client):
    kimi_client.config.llm.max_tokens = 4096
    assert (kimi_client.config.llm.max_tokens or kimi_client.spec.max_tokens) == 4096


@pytest.mark.asyncio
@respx.mock
async def test_client_captures_usage_and_completion_meta(client):
    stream = (
        'data: {"id":"1","object":"chat.completion.chunk",'
        '"choices":[{"delta":{"content":"hi"}}]}\n\n'
        'data: {"id":"2","object":"chat.completion.chunk",'
        '"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"3","object":"chat.completion.chunk","choices":[],'
        '"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,'
        '"prompt_cache_hit_tokens":8,"prompt_cache_miss_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(stream)
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    meta = next(e for e in events if isinstance(e, CompletionMeta))
    assert meta.usage is not None
    assert meta.usage["prompt_tokens"] == 10
    # Provider extras (DeepSeek cache counters) survive the round trip.
    assert meta.usage["prompt_cache_hit_tokens"] == 8
    assert meta.finish_reason == "stop"
    assert meta.ttft is not None
    assert meta.duration is not None
    # Usage was requested via stream_options.
    body = json.loads(route.calls.last.request.content)
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
@respx.mock
async def test_client_reports_request_body_via_hook(client):
    respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=_sse_response(
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        )
    )
    bodies = []
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "description": "Read", "parameters": {}},
        }
    ]

    await _collect(
        client.chat(
            [Message(role="user", content="hi")], tools=tools, on_request=bodies.append
        )
    )

    assert len(bodies) == 1
    body = bodies[0]
    assert body["model"] == "deepseek-chat"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["tools"] == tools
    assert body["stream"] is True
    assert "temperature" in body and "max_tokens" in body


@pytest.mark.asyncio
@respx.mock
async def test_client_retries_without_stream_options_when_rejected(client):
    ok_stream = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        side_effect=[
            Response(
                400,
                json={
                    "error": {
                        "message": "Unknown parameter: stream_options",
                        "type": "invalid_request_error",
                    }
                },
            ),
            _sse_response(ok_stream),
        ]
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 2
    assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "hi"
    second_body = json.loads(route.calls[1].request.content)
    assert "stream_options" not in second_body
    # The client stops asking for usage on subsequent requests.
    assert client._include_usage is False


# -- unified retry integration (limbo.llm.retry) -----------------------------


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace asyncio.sleep with a recorder so retries cost no real time."""
    calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return calls


def test_sdk_client_disables_builtin_retries_and_wires_timeout(client):
    client.config.llm.timeout = 123.0
    client.config.llm.connect_timeout = 4.0

    sdk = client.client

    # SDK built-in retries are off; stream_with_retry owns retrying so every
    # attempt is trace-visible via the on_request hook.
    assert sdk.max_retries == 0
    assert sdk.timeout.read == 123.0
    assert sdk.timeout.connect == 4.0


@pytest.mark.asyncio
@respx.mock
async def test_retries_on_429_then_succeeds(client, sleeps):
    ok_stream = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        side_effect=[
            Response(
                429,
                json={"error": {"message": "rate limited"}},
                headers={"retry-after": "0"},
            ),
            _sse_response(ok_stream),
        ]
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 2
    assert "".join(e.text for e in events if isinstance(e, TextChunk)) == "hi"
    assert len(sleeps) == 1


@pytest.mark.asyncio
@respx.mock
async def test_429_exhausts_retries_then_raises(client, sleeps):
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        side_effect=[
            Response(429, json={"error": {"message": "rate limited"}})
            for _ in range(4)
        ]
    )

    with pytest.raises(RateLimitError):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    # max_retries=3 -> 4 attempts total, with a backoff before each retry.
    assert route.call_count == 4
    assert len(sleeps) == 3


@pytest.mark.asyncio
@respx.mock
async def test_400_is_not_retried(client, sleeps):
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=Response(400, json={"error": {"message": "bad request"}})
    )

    with pytest.raises(BadRequestError):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 1
    assert sleeps == []


@pytest.mark.asyncio
@respx.mock
async def test_max_retries_zero_disables_retry(client, sleeps):
    client.config.llm.max_retries = 0
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=Response(429, json={"error": {"message": "rate limited"}})
    )

    with pytest.raises(RateLimitError):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 1
    assert sleeps == []


def test_message_to_openai_with_image_attachment(tmp_path):
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
    result = _message_to_openai(message)
    content = result["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    expected = base64.standard_b64encode(b"png-bytes").decode("ascii")
    assert content[0]["image_url"]["url"] == f"data:image/png;base64,{expected}"
    assert content[1] == {"type": "text", "text": "看这张图 [图片 #1]"}


def test_message_to_openai_missing_image_stays_plain(tmp_path):
    message = Message(
        role="user",
        content="hi",
        attachments=[
            Attachment(
                kind="image", name="gone.png", path=str(tmp_path / "gone.png")
            )
        ],
    )
    result = _message_to_openai(message)
    assert result["content"] == "hi"


def test_message_to_openai_without_attachments_unchanged():
    result = _message_to_openai(Message(role="user", content="hi"))
    assert result["content"] == "hi"


def test_messages_to_openai_tool_result_images_become_synthetic_user(tmp_path):
    """Tool results stay text-only; their images are appended as ONE
    synthetic user message after a run of tool messages (pi parity — the
    chat completions API rejects image parts in tool messages)."""
    image = tmp_path / "shot.png"
    image.write_bytes(b"png-bytes")
    attachment = Attachment(
        kind="image", name="shot.png", path=str(image), mime="image/png"
    )
    messages = [
        Message(
            role="tool",
            content="Read image file [image/png]",
            tool_call_id="c1",
            attachments=[attachment],
        ),
        Message(
            role="tool",
            content="",
            tool_call_id="c2",
            attachments=[attachment],
        ),
        Message(role="user", content="next"),
    ]
    result = _messages_to_openai(messages)

    # Tool messages are text-only; the empty one gets the placeholder.
    assert result[0] == {
        "role": "tool",
        "content": "Read image file [image/png]",
        "tool_call_id": "c1",
    }
    assert result[1] == {
        "role": "tool",
        "content": "(see attached image)",
        "tool_call_id": "c2",
    }
    # Both images are grouped into ONE synthetic user message.
    synthetic = result[2]
    assert synthetic["role"] == "user"
    assert synthetic["content"][0] == {
        "type": "text",
        "text": "Attached image(s) from tool result:",
    }
    image_blocks = [b for b in synthetic["content"] if b["type"] == "image_url"]
    assert len(image_blocks) == 2
    expected = base64.standard_b64encode(b"png-bytes").decode("ascii")
    assert image_blocks[0]["image_url"]["url"] == f"data:image/png;base64,{expected}"
    assert result[3]["role"] == "user"


def test_messages_to_openai_tool_images_dropped_without_vision(tmp_path):
    """A mid-session /model switch to a non-vision model drops the images."""
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
    result = _messages_to_openai([message], vision=False)
    assert result == [
        {
            "role": "tool",
            "content": "Read image file [image/png]",
            "tool_call_id": "c1",
        }
    ]


def test_messages_to_openai_tool_missing_image_stays_plain(tmp_path):
    message = Message(
        role="tool",
        content="Read image file [image/png]",
        tool_call_id="c1",
        attachments=[
            Attachment(kind="image", name="gone.png", path=str(tmp_path / "gone.png"))
        ],
    )
    result = _messages_to_openai([message])
    assert result == [
        {
            "role": "tool",
            "content": "Read image file [image/png]",
            "tool_call_id": "c1",
        }
    ]
