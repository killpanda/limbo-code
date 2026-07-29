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
from limbo.models import LLMEvent, Message, TextChunk, ThinkingChunk, ToolCallEvent

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
                timeout=httpx.Timeout(600.0, connect=30.0),
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

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMEvent]:
        body = self._build_body(messages, tools)
        async with self.client.stream("POST", "/v1/messages", json=body) as resp:
            if resp.status_code != 200:
                error_body = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"Error code: {resp.status_code} - {error_body}")

            # Tool-use blocks accumulate streamed partial JSON by index.
            tool_blocks: dict[int, dict[str, Any]] = {}
            async for event in _iter_sse(resp):
                event_type = event.get("type")
                if event_type == "content_block_start":
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
                    raise RuntimeError(
                        f"{error.get('type', 'error')}: "
                        f"{error.get('message', 'unknown stream error')}"
                    )

            # A truncated stream may leave tool blocks open; flush them so the
            # agent loop can surface the (possibly partial) call.
            for index in sorted(tool_blocks):
                yield _tool_call_event(tool_blocks[index])


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
