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
    _tool_to_responses,
)
from limbo.llm.retry import LLMHttpError
from limbo.models import (
    Attachment,
    CompletionMeta,
    Message,
    TextChunk,
    ThinkingChunk,
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


def _text_stream(text: str = "hello") -> Response:
    return _sse(
        {"type": "response.output_text.delta", "delta": text},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 64},
                    "output_tokens_details": {"reasoning_tokens": 10},
                },
            },
        },
    )


# -- request shape -----------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_request_headers_and_body(client):
    client.config.llm.thinking_effort = "high"
    route = respx.post(RESPONSES_URL).mock(return_value=_text_stream())

    await _collect(
        client.chat(
            [
                Message(role="system", content="be terse"),
                Message(role="user", content="hi"),
            ],
            tools=[],
        )
    )

    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer relay-key"
    body = json.loads(request.content)
    assert body["model"] == "gpt-5.5"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["instructions"] == "be terse"
    assert body["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
    assert body["reasoning"] == {"effort": "high", "summary": "auto"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["max_output_tokens"] == 128_000
    assert body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is True
    # Reasoning models reject an explicit temperature.
    assert "temperature" not in body


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_model_omits_temperature_by_default(client):
    # pi only sends temperature when the user sets it explicitly; limbo's
    # config always carries a default, so reasoning models must not get it
    # even with thinking_effort unset (official endpoints 400 otherwise).
    route = respx.post(RESPONSES_URL).mock(return_value=_text_stream())

    await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert "reasoning" not in body
    assert "temperature" not in body
    # include is still requested: the model reasons regardless of the switch.
    assert body["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
@respx.mock
async def test_unsupported_effort_warns_and_omits_reasoning(client):
    client.config.llm.thinking_effort = "max"  # gpt-5.5 tops out at xhigh
    route = respx.post(RESPONSES_URL).mock(return_value=_text_stream())

    with pytest.warns(UserWarning, match="not supported"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    body = json.loads(route.calls.last.request.content)
    assert "reasoning" not in body


@pytest.mark.asyncio
@respx.mock
async def test_tools_flattened(client):
    route = respx.post(RESPONSES_URL).mock(return_value=_text_stream())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]

    await _collect(client.chat([Message(role="user", content="hi")], tools=tools))

    body = json.loads(route.calls.last.request.content)
    assert body["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_tool_to_responses_shape():
    assert _tool_to_responses(
        {"type": "function", "function": {"name": "ls", "description": "List"}}
    ) == {
        "type": "function",
        "name": "ls",
        "description": "List",
        "parameters": {"type": "object"},
    }


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


def test_foreign_signature_is_not_replayed():
    # An Anthropic-dialect signature must not leak into Responses input.
    message = Message(
        role="assistant",
        content="hi",
        reasoning="thoughts",
        reasoning_signature="sig_anthropic",
    )
    _, items = _messages_to_responses([message])
    assert all(item.get("type") != "reasoning" for item in items)


def test_user_message_with_image_attachment(tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG")
    message = Message(
        role="user",
        content="what is this",
        attachments=[
            Attachment(kind="image", name="shot.png", path=str(image), mime="image/png")
        ],
    )
    _, items = _messages_to_responses([message])
    content = items[0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "input_text", "text": "what is this"}


def test_user_message_missing_image_degrades_to_text(tmp_path):
    message = Message(
        role="user",
        content="note",
        attachments=[
            Attachment(
                kind="image",
                name="gone.png",
                path=str(tmp_path / "gone.png"),
                mime="image/png",
            )
        ],
    )
    _, items = _messages_to_responses([message])
    assert items[0]["content"] == [{"type": "input_text", "text": "note"}]


# -- stream events -----------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_text_and_usage_events(client):
    respx.post(RESPONSES_URL).mock(return_value=_text_stream("hi there"))

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    text = "".join(e.text for e in events if isinstance(e, TextChunk))
    assert text == "hi there"
    meta = next(e for e in events if isinstance(e, CompletionMeta))
    assert meta.usage["input_tokens"] == 100
    assert meta.usage["input_tokens_details"] == {"cached_tokens": 64}
    assert meta.finish_reason == "completed"
    assert meta.ttft is not None and meta.duration is not None


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_summary_streams_as_thinking(client):
    respx.post(RESPONSES_URL).mock(
        return_value=_sse(
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking..."},
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed", "response": {"status": "completed"}},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    thinking = "".join(e.text for e in events if isinstance(e, ThinkingChunk))
    assert thinking == "thinking..."


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_item_captured_as_prefixed_signature(client):
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "enc_payload",
        "summary": [],
    }
    respx.post(RESPONSES_URL).mock(
        return_value=_sse(
            {"type": "response.output_item.done", "item": reasoning_item},
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": {"status": "completed"}},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    signatures = [
        e.signature for e in events if isinstance(e, ThinkingChunk) and e.signature
    ]
    assert signatures == [RESPONSES_SIGNATURE_PREFIX + json.dumps(reasoning_item)]


@pytest.mark.asyncio
@respx.mock
async def test_reasoning_item_backfilled_from_completed_response(client):
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_9",
        "encrypted_content": "enc_payload",
    }
    respx.post(RESPONSES_URL).mock(
        return_value=_sse(
            {"type": "response.output_text.delta", "delta": "ok"},
            {
                "type": "response.completed",
                "response": {"status": "completed", "output": [reasoning_item]},
            },
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    signatures = [
        e.signature for e in events if isinstance(e, ThinkingChunk) and e.signature
    ]
    assert signatures == [RESPONSES_SIGNATURE_PREFIX + json.dumps(reasoning_item)]


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


@pytest.mark.asyncio
@respx.mock
async def test_truncated_stream_flushes_open_tool_call(client):
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
                "delta": '{"path": "a.py"}',
            },
            {"type": "response.completed", "response": {"status": "completed"}},
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].arguments == {"path": "a.py"}


@pytest.mark.asyncio
@respx.mock
async def test_incomplete_status_reports_reason(client):
    respx.post(RESPONSES_URL).mock(
        return_value=_sse(
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            },
        )
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    meta = next(e for e in events if isinstance(e, CompletionMeta))
    assert meta.finish_reason == "max_output_tokens"


@pytest.mark.asyncio
@respx.mock
async def test_response_failed_raises(client):
    respx.post(RESPONSES_URL).mock(
        return_value=_sse(
            {
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "boom"},
                },
            },
        )
    )

    with pytest.raises(RuntimeError, match="server_error: boom"):
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))


@pytest.mark.asyncio
@respx.mock
async def test_rate_limit_stream_error_is_retryable(client, monkeypatch):
    monkeypatch.setattr("limbo.llm.retry.asyncio.sleep", _no_sleep)
    client.config.llm.max_retries = 1
    route = respx.post(RESPONSES_URL).mock(
        side_effect=[
            _sse(
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "slow down",
                        },
                    },
                },
            ),
            _text_stream("recovered"),
        ]
    )

    events = await _collect(client.chat([Message(role="user", content="hi")], tools=[]))

    assert route.call_count == 2
    text = "".join(e.text for e in events if isinstance(e, TextChunk))
    assert text == "recovered"


async def _no_sleep(delay):
    return None


@pytest.mark.asyncio
@respx.mock
async def test_http_error_raises_llm_http_error(client):
    client.config.llm.max_retries = 0
    respx.post(RESPONSES_URL).mock(
        return_value=Response(400, text='{"error": {"message": "bad request"}}')
    )

    with pytest.raises(LLMHttpError) as exc_info:
        await _collect(client.chat([Message(role="user", content="hi")], tools=[]))
    assert exc_info.value.status_code == 400


# -- client plumbing ---------------------------------------------------------


def test_missing_api_key_raises_with_env_hint():
    import os

    os.environ.pop("CODEX_API_KEY", None)
    cfg = Config()
    cfg.llm.model = "gpt-5.5"
    cfg.providers["codex"] = ProviderOverride(api_key_env="MY_CODEX_KEY")
    os.environ.pop("MY_CODEX_KEY", None)
    c = OpenAIResponsesClient(cfg)
    with pytest.raises(ValueError, match="MY_CODEX_KEY"):
        _ = c.client


def test_catalog_resolves_codex_models():
    from limbo.llm.catalog import resolve_model

    # gpt-5.5: verified live against the target relay (1M context,
    # vision, 128K output per the relay's /models).
    spec = resolve_model("gpt-5.5")
    assert spec.provider.id == "codex"
    assert spec.provider.api == "openai-responses"
    assert spec.context_window == 1_000_000
    assert spec.max_tokens == 128_000
    assert spec.vision is True
    assert spec.thinking_format == "openai-responses"
    assert spec.thinking_levels == {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
    }

    # pi-derived entries (unverified against the target relay).
    sol = resolve_model("gpt-5.6-sol")
    assert sol.context_window == 272_000
    assert sol.thinking_levels["max"] == "max"
    assert sol.vision is True

    spark = resolve_model("gpt-5.3-codex-spark")
    assert spark.context_window == 128_000
    assert spark.vision is False
    assert "max" not in spark.thinking_levels

    for model_id in ("gpt-5.4", "gpt-5.4-mini", "gpt-5.6-terra", "gpt-5.6-luna"):
        assert resolve_model(model_id).provider is spec.provider, model_id


@pytest.mark.asyncio
async def test_close_releases_http_client(client):
    _ = client.client
    await client.close()
    assert client._client is None
