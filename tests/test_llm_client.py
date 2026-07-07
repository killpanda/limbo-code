import json
from unittest.mock import AsyncMock

import pytest
import respx
from httpx import Response

from limbo.config import Config
from limbo.llm.openai_client import OpenAICompatibleClient, _message_to_openai
from limbo.models import Message, ToolCallEvent


@pytest.fixture
def client():
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.base_url = "https://api.test.com/v1"
    return OpenAICompatibleClient(cfg)


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
