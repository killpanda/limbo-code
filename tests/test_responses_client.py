"""Tests for the OpenAI Responses API client (codex dialect)."""

import json

import pytest
import respx
from httpx import Response

from limbo.config import Config, ProviderOverride
from limbo.llm.client import RESPONSES_SIGNATURE_PREFIX
from limbo.llm.responses_client import (
    OpenAIResponsesClient,
    _messages_to_responses,
)
from limbo.models import (
    Message,
    ToolCallEvent,
)

RESPONSES_URL = "https://relay.example.com/v1/responses"


@pytest.fixture
def client():
    cfg = Config()
    cfg.llm.model = "gpt-5.5"
    cfg.providers["codex"] = ProviderOverride(
        base_url="https://relay.example.com/v1", api_key="relay-key"
    )
    return OpenAIResponsesClient(cfg)


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


# -- message conversion ------------------------------------------------------


def test_messages_to_responses_full_conversation():
    reasoning_item = {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"}
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hello"),
        Message(
            role="assistant",
            content="working",
            reasoning="thoughts",
            reasoning_signature=RESPONSES_SIGNATURE_PREFIX + json.dumps(reasoning_item),
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": {"path": "a.py"}},
                }
            ],
        ),
        Message(role="tool", content="file body", tool_call_id="call_1"),
    ]

    instructions, items = _messages_to_responses(messages)

    assert instructions == "sys"
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        reasoning_item,  # replayed verbatim
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "working", "annotations": []}
            ],
            "status": "completed",
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read",
            "arguments": '{"path": "a.py"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "file body"},
    ]


# -- stream events -----------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_function_call_streams_and_completes(client):
    respx.post(RESPONSES_URL).mock(
        return_value=_sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"path":',
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": ' "a.py"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "arguments": '{"path": "a.py"}',
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": '{"path": "a.py"}',
                },
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "read"
    assert calls[0].arguments == {"path": "a.py"}


# -- client plumbing ---------------------------------------------------------


