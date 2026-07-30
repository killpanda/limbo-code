"""OpenAI Responses API client (e.g. Codex models via a relay).

Implemented directly on httpx SSE (like the Anthropic client) rather than
the OpenAI SDK so the request body stays explicit — the Responses dialect
differs from chat completions in message items, tool shape, and stream
event names. Conventions follow pi's openai-codex-responses provider,
trimmed to the API-key relay scenario (no OAuth, no chatgpt.com backend
headers):

- Auth is ``Authorization: Bearer <api_key>``; requests go to
  ``POST {base_url}/responses``.
- ``store: false`` + ``stream: true``; system messages become the top-level
  ``instructions`` string.
- Reasoning models get ``reasoning: {effort, summary: "auto"}`` and ask for
  ``include: ["reasoning.encrypted_content"]``; returned reasoning items
  are captured as the message signature (dialect-prefixed, see
  ``limbo.llm.client``) so they can be replayed on later turns.
"""

from __future__ import annotations

import base64
import json
import time
import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from limbo.config import Config
from limbo.llm.catalog import (
    ModelSpec,
    resolve_api_key,
    resolve_api_key_env,
    resolve_base_url,
    resolve_headers,
    resolve_model,
)
from limbo.llm.client import RESPONSES_SIGNATURE_PREFIX, RequestHook
from limbo.llm.retry import (
    LLMHttpError,
    LLMOverloadedError,
    RetryPolicy,
    parse_retry_after,
    stream_with_retry,
)
from limbo.llm.sse import iter_sse
from limbo.models import (
    CompletionMeta,
    LLMEvent,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
)


class OpenAIResponsesClient:
    """Client for the OpenAI Responses API with SSE streaming."""

    def __init__(self, config: Config):
        self.config = config
        self.spec: ModelSpec = resolve_model(config.llm.model)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = resolve_api_key(self.spec, self.config)
            if not api_key:
                env = resolve_api_key_env(self.spec, self.config)
                raise ValueError(
                    f"No API key for provider {self.spec.provider.id!r}. "
                    f"Set [llm] api_key in ~/.limbo/config.toml"
                    + (f" or the ${env} environment variable." if env else ".")
                )
            self._client = httpx.AsyncClient(
                base_url=resolve_base_url(self.spec, self.config).rstrip("/"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                    **resolve_headers(self.spec, self.config),
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

    def _reasoning_params(self) -> dict[str, Any]:
        """reasoning: {effort, summary} for the openai-responses dialect."""
        spec = self.spec
        if not spec.reasoning or spec.thinking_format != "openai-responses":
            return {}
        effort = self.config.llm.thinking_effort
        if effort is None:
            return {}
        value = spec.thinking_levels.get(effort)
        if value is None:
            warnings.warn(
                f"thinking_effort={effort!r} is not supported by {spec.id} "
                f"(supported: {sorted(spec.thinking_levels)}); ignoring.",
                stacklevel=2,
            )
            return {}
        return {"reasoning": {"effort": value, "summary": "auto"}}

    def _build_body(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        instructions, input_items = _messages_to_responses(messages)
        body: dict[str, Any] = {
            "model": self.config.llm.model,
            "store": False,
            "stream": True,
            "input": input_items,
            "max_output_tokens": self.config.llm.max_tokens or self.spec.max_tokens,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if instructions:
            body["instructions"] = instructions
        body.update(self._reasoning_params())
        if self.spec.reasoning:
            # Ask for replayable reasoning items (needed for function-call
            # pairing under store:false); only reasoning models emit them.
            body["include"] = ["reasoning.encrypted_content"]
        if "reasoning" not in body:
            # Reasoning models reject an explicit temperature.
            body["temperature"] = self.config.llm.temperature
        if tools:
            body["tools"] = [_tool_to_responses(t) for t in tools]
        # Provider extra-body fields (quirks not covered above); tool-scoped
        # extras apply only when the request carries tools.
        body.update(self.spec.provider.extra_body)
        if tools:
            body.update(self.spec.provider.tool_extra_body)
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
        reasoning_signature: str | None = None
        async with self.client.stream("POST", "/responses", json=body) as resp:
            if resp.status_code != 200:
                error_body = (await resp.aread()).decode("utf-8", "replace")
                raise LLMHttpError(
                    status_code=resp.status_code,
                    message=f"Error code: {resp.status_code} - {error_body}",
                    retry_after=parse_retry_after(resp.headers.get("retry-after")),
                )

            # function_call argument fragments accumulate by item id.
            tool_calls: dict[str, dict[str, Any]] = {}
            async for event in iter_sse(resp):
                if ttft is None:
                    ttft = time.monotonic() - start
                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    yield TextChunk(text=event.get("delta", ""))
                elif event_type in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    yield ThinkingChunk(text=event.get("delta", ""))
                elif event_type == "response.output_item.added":
                    item = event.get("item", {})
                    if item.get("type") == "function_call":
                        tool_calls[item.get("id", "")] = {
                            "id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "json": "",
                        }
                elif event_type == "response.function_call_arguments.delta":
                    call = tool_calls.get(event.get("item_id", ""))
                    if call is not None:
                        call["json"] += event.get("delta", "")
                elif event_type == "response.function_call_arguments.done":
                    call = tool_calls.get(event.get("item_id", ""))
                    if call is not None:
                        call["json"] = event.get("arguments", call["json"])
                elif event_type == "response.output_item.done":
                    item = event.get("item", {})
                    item_type = item.get("type")
                    if item_type == "function_call":
                        call = tool_calls.pop(item.get("id", ""), None) or {
                            "id": item.get("call_id", ""),
                            "name": item.get("name", ""),
                            "json": item.get("arguments", ""),
                        }
                        yield _tool_call_event(call)
                    elif item_type == "reasoning":
                        signature = _reasoning_signature(item)
                        if signature is not None:
                            reasoning_signature = signature
                            yield ThinkingChunk(text="", signature=signature)
                elif event_type in ("response.completed", "response.incomplete"):
                    response = event.get("response", {})
                    event_usage = response.get("usage")
                    if isinstance(event_usage, dict):
                        usage.update(event_usage)
                    finish_reason = response.get("status")
                    if finish_reason == "incomplete":
                        details = response.get("incomplete_details") or {}
                        finish_reason = details.get("reason", "incomplete")
                    if reasoning_signature is None:
                        # Some backends only attach reasoning items to the
                        # final response.output; backfill the signature.
                        for output_item in response.get("output") or []:
                            if output_item.get("type") == "reasoning":
                                signature = _reasoning_signature(output_item)
                                if signature is not None:
                                    reasoning_signature = signature
                                    yield ThinkingChunk(
                                        text="", signature=signature
                                    )
                                break
                elif event_type == "response.failed":
                    error = event.get("response", {}).get("error") or {}
                    _raise_stream_error(error)
                elif event_type == "error":
                    _raise_stream_error(event)

            # A truncated stream may leave tool calls open; flush them so the
            # agent loop can surface the (possibly partial) call.
            for item_id in sorted(tool_calls):
                yield _tool_call_event(tool_calls[item_id])

        yield CompletionMeta(
            usage=usage or None,
            finish_reason=finish_reason,
            ttft=ttft,
            duration=time.monotonic() - start,
        )


def _raise_stream_error(error: dict[str, Any]) -> None:
    code = error.get("code") or "error"
    message = error.get("message") or "unknown stream error"
    if isinstance(code, str) and ("overload" in code or "rate_limit" in code):
        # Transient provider pressure; retryable while no event has been
        # yielded yet (see llm.retry).
        raise LLMOverloadedError(f"{code}: {message}")
    raise RuntimeError(f"{code}: {message}")


def _reasoning_signature(item: dict[str, Any]) -> str | None:
    """Serialize a reasoning item for replay, with the dialect prefix.

    Without ``encrypted_content`` the item carries no replayable state
    (store:false backends need it), so nothing is stored.
    """
    if not item.get("encrypted_content"):
        return None
    return RESPONSES_SIGNATURE_PREFIX + json.dumps(item)


def _tool_call_event(call: dict[str, Any]) -> ToolCallEvent:
    raw = call["json"]
    try:
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        args = {"raw_arguments": raw, "parse_error": str(e)}
    return ToolCallEvent(id=call["id"], name=call["name"], arguments=args)


def _tool_to_responses(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI function tool definition to the flat Responses shape."""
    function = tool.get("function", tool)
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object"}),
    }


def _messages_to_responses(
    messages: list[Message],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert internal messages to Responses ``instructions`` + ``input`` items."""
    system_parts: list[str] = []
    items: list[dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
        elif message.role == "user":
            items.append({"role": "user", "content": _user_content(message)})
        elif message.role == "assistant":
            items.extend(_assistant_items(message))
        elif message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id or "",
                    "output": message.content or "",
                }
            )

    return "\n\n".join(system_parts), items


def _user_content(message: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if message.attachments:
        # Multimodal user message: input_image blocks + text. Missing files
        # (expired session attachments) are skipped.
        for attachment in message.attachments:
            if attachment.kind != "image":
                continue
            try:
                data = base64.standard_b64encode(
                    Path(attachment.path).read_bytes()
                ).decode("ascii")
            except OSError:
                continue
            mime = attachment.mime or "image/png"
            blocks.append(
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{mime};base64,{data}",
                }
            )
    blocks.append({"type": "input_text", "text": message.content or ""})
    return blocks


def _assistant_items(message: Message) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    signature = message.reasoning_signature or ""
    if signature.startswith(RESPONSES_SIGNATURE_PREFIX):
        # Replay the stored reasoning item verbatim (function-call pairing
        # under store:false). Foreign-dialect signatures are skipped.
        try:
            items.append(json.loads(signature[len(RESPONSES_SIGNATURE_PREFIX) :]))
        except json.JSONDecodeError:
            pass
    if message.content:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": message.content,
                        "annotations": [],
                    }
                ],
                "status": "completed",
            }
        )
    for tc in message.tool_calls or []:
        function = tc.get("function", {})
        arguments = function.get("arguments", {})
        items.append(
            {
                "type": "function_call",
                "call_id": tc.get("id", ""),
                "name": function.get("name", ""),
                "arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments)
                ),
            }
        )
    return items
