"""Anthropic Messages API client (e.g. Kimi For Coding).

Implemented directly on httpx SSE rather than the anthropic SDK so that
newer dialect features — adaptive thinking (``thinking: {type: adaptive}``
plus ``output_config.effort``), thinking signatures, and interleaved
thinking deltas — work regardless of the installed SDK version.

Conventions follow pi's anthropic-messages provider:

- API keys authenticate via the ``x-api-key`` header.
- Reasoning models use adaptive thinking; ``temperature`` is omitted while
  thinking is enabled.
- Assistant messages replay their thinking block first, carrying the stored
  signature (Kimi accepts empty signatures).
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import AsyncIterator
from typing import Any

import httpx

from limbo.config import Config
from limbo.llm.catalog import (
    ModelSpec,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from limbo.llm.client import RequestHook
from limbo.llm.retry import (
    LLMHttpError,
    LLMOverloadedError,
    RetryPolicy,
    parse_retry_after,
    stream_with_retry,
)
from limbo.models import (
    CompletionMeta,
    LLMEvent,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
)

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicMessagesClient:
    """Client for the Anthropic Messages API with SSE streaming."""

    def __init__(self, config: Config):
        self.config = config
        self.spec: ModelSpec = resolve_model(config.llm.model)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = resolve_api_key(self.spec, self.config.llm.api_key)
            if not api_key:
                env = self.spec.provider.api_key_env
                raise ValueError(
                    f"No API key for provider {self.spec.provider.id!r}. "
                    f"Set [llm] api_key in ~/.limbo/config.toml"
                    + (f" or the ${env} environment variable." if env else ".")
                )
            self._client = httpx.AsyncClient(
                base_url=resolve_base_url(self.spec, self.config.llm.base_url).rstrip("/"),
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "accept": "application/json",
                    **self.spec.provider.headers,
                },
                timeout=httpx.Timeout(
                    self.config.llm.timeout,
                    connect=self.config.llm.connect_timeout,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if it has been created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- request building ----------------------------------------------------

    def _thinking_params(self) -> dict[str, Any]:
        """Adaptive thinking config for reasoning models."""
        spec = self.spec
        if not spec.reasoning or spec.thinking_format != "anthropic-adaptive":
            return {}
        params: dict[str, Any] = {
            "thinking": {"type": "adaptive", "display": "summarized"}
        }
        effort = self.config.llm.thinking_effort
        if effort is not None:
            value = spec.thinking_levels.get(effort)
            if value is None:
                warnings.warn(
                    f"thinking_effort={effort!r} is not supported by {spec.id} "
                    f"(supported: {sorted(spec.thinking_levels)}); ignoring.",
                    stacklevel=2,
                )
            else:
                params["output_config"] = {"effort": value}
        return params

    def _build_body(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        system, request_messages = _messages_to_anthropic(messages)
        body: dict[str, Any] = {
            "model": self.config.llm.model,
            "max_tokens": self.config.llm.max_tokens or self.spec.max_tokens,
            "messages": request_messages,
            "stream": True,
        }
        if system:
            body["system"] = system
        body.update(self._thinking_params())
        if "thinking" not in body:
            # Temperature is incompatible with extended thinking.
            body["temperature"] = self.config.llm.temperature
        if tools:
            body["tools"] = [_tool_to_anthropic(t) for t in tools]
        return body

    # -- streaming -----------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_request: RequestHook | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream events, retrying pre-first-event failures (see llm.retry)."""
        policy = RetryPolicy.from_config(self.config.llm)
        return stream_with_retry(
            lambda: self._chat_inner(messages, tools, on_request), policy
        )

    async def _chat_inner(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_request: RequestHook | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """One streaming attempt: build the request and yield its events."""
        body = self._build_body(messages, tools)
        if on_request is not None:
            on_request(body)
        start = time.monotonic()
        ttft: float | None = None
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        async with self.client.stream("POST", "/v1/messages", json=body) as resp:
            if resp.status_code != 200:
                error_body = (await resp.aread()).decode("utf-8", "replace")
                raise LLMHttpError(
                    status_code=resp.status_code,
                    message=f"Error code: {resp.status_code} - {error_body}",
                    retry_after=parse_retry_after(resp.headers.get("retry-after")),
                )

            # Tool-use blocks accumulate streamed partial JSON by index.
            tool_blocks: dict[int, dict[str, Any]] = {}
            async for event in _iter_sse(resp):
                if ttft is None:
                    ttft = time.monotonic() - start
                event_type = event.get("type")
                if event_type == "message_start":
                    # usage here carries input tokens, incl. Anthropic cache
                    # counters (cache_read_input_tokens = cache hits).
                    message_usage = event.get("message", {}).get("usage")
                    if isinstance(message_usage, dict):
                        usage.update(message_usage)
                elif event_type == "message_delta":
                    delta_usage = event.get("usage")
                    if isinstance(delta_usage, dict):
                        usage.update(delta_usage)
                    stop = event.get("delta", {}).get("stop_reason")
                    if stop:
                        finish_reason = stop
                elif event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_blocks[event["index"]] = {
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "json": "",
                        }
                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type")
                    if delta_type == "thinking_delta":
                        yield ThinkingChunk(text=delta.get("thinking", ""))
                    elif delta_type == "signature_delta":
                        yield ThinkingChunk(
                            text="", signature=delta.get("signature", "")
                        )
                    elif delta_type == "text_delta":
                        yield TextChunk(text=delta.get("text", ""))
                    elif delta_type == "input_json_delta":
                        block = tool_blocks.get(event["index"])
                        if block is not None:
                            block["json"] += delta.get("partial_json", "")
                elif event_type == "content_block_stop":
                    block = tool_blocks.pop(event["index"], None)
                    if block is not None:
                        yield _tool_call_event(block)
                elif event_type == "error":
                    error = event.get("error", {})
                    error_type = error.get("type", "error")
                    error_message = error.get("message", "unknown stream error")
                    if error_type == "overloaded_error":
                        # Provider overload is transient; retryable while no
                        # event has been yielded yet.
                        raise LLMOverloadedError(
                            f"{error_type}: {error_message}"
                        )
                    raise RuntimeError(f"{error_type}: {error_message}")

            # A truncated stream may leave tool blocks open; flush them so the
            # agent loop can surface the (possibly partial) call.
            for index in sorted(tool_blocks):
                yield _tool_call_event(tool_blocks[index])

        yield CompletionMeta(
            usage=usage or None,
            finish_reason=finish_reason,
            ttft=ttft,
            duration=time.monotonic() - start,
        )


def _tool_call_event(block: dict[str, Any]) -> ToolCallEvent:
    raw = block["json"]
    try:
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        args = {"raw_arguments": raw, "parse_error": str(e)}
    return ToolCallEvent(id=block["id"], name=block["name"], arguments=args)


async def _iter_sse(resp: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Yield parsed ``data:`` payloads from an Anthropic SSE stream."""
    data_lines: list[str] = []
    async for line in resp.aiter_lines():
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        # event:/id:/retry: lines and comments (: ping) are ignored; the
        # payload's own "type" field drives dispatch.


def _tool_to_anthropic(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI function tool definition to Anthropic's shape."""
    function = tool.get("function", tool)
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {"type": "object"}),
    }


def _messages_to_anthropic(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert internal messages to Anthropic's system/messages split.

    System messages are joined into the top-level ``system`` parameter.
    Consecutive tool results are merged into a single user message, as the
    API requires tool_result blocks inside a user turn.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
        elif message.role == "user":
            converted.append({"role": "user", "content": message.content or ""})
        elif message.role == "assistant":
            converted.append(
                {"role": "assistant", "content": _assistant_blocks(message)}
            )
        elif message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content or "",
            }
            previous = converted[-1] if converted else None
            if previous is not None and previous.get("role") == "user" and isinstance(
                previous.get("content"), list
            ):
                previous["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})

    return "\n\n".join(system_parts), converted


def _assistant_blocks(message: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if message.reasoning:
        # Kimi accepts empty signatures on replayed thinking blocks; real
        # Anthropic models get the stored signature.
        blocks.append(
            {
                "type": "thinking",
                "thinking": message.reasoning,
                "signature": message.reasoning_signature or "",
            }
        )
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for tc in message.tool_calls or []:
        function = tc.get("function", {})
        arguments = function.get("arguments", {})
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": function.get("name", ""),
                "input": (
                    arguments
                    if isinstance(arguments, dict)
                    else json.loads(arguments or "{}")
                ),
            }
        )
    # The API rejects assistant messages with no content blocks.
    return blocks or [{"type": "text", "text": ""}]
