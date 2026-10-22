import asyncio

import pytest
import respx
from httpx import Response

from limbo.config import Config
from limbo.llm.openai_client import (
    OpenAICompatibleClient,
    _message_to_openai,
)
from limbo.models import (
    Message,
    ToolCallEvent,
)


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


# -- model-specific behavior (catalog-driven) --------------------------------


def test_message_to_openai_replays_reasoning_content_when_required():
    message = Message(role="assistant", content="done", reasoning="my thoughts")
    result = _message_to_openai(message, include_reasoning=True)
    assert result["reasoning_content"] == "my thoughts"


# -- unified retry integration (limbo.llm.retry) -----------------------------


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace asyncio.sleep with a recorder so retries cost no real time."""
    calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return calls


