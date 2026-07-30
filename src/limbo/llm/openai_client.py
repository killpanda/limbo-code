"""OpenAI-compatible LLM client."""

from __future__ import annotations

import base64
import json
import time
import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI, BadRequestError

from limbo.config import Config
from limbo.llm.catalog import (
    ModelSpec,
    resolve_api_key,
    resolve_base_url,
    resolve_headers,
    resolve_model,
)
from limbo.llm.client import RequestHook
from limbo.llm.retry import RetryPolicy, stream_with_retry
from limbo.models import (
    CompletionMeta,
    LLMEvent,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
)


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat completions with streaming.

    Adapts request parameters and stream parsing to the model's catalog spec
    (see ``limbo.llm.catalog``): per-model ``max_tokens``, thinking-parameter
    format (``reasoning_effort`` vs ``thinking: {type: ...}``), and
    ``reasoning_content`` deltas/replay for reasoning models such as Kimi K3.
    """

    def __init__(self, config: Config):
        self.config = config
        self.spec: ModelSpec = resolve_model(config.llm.model)
        self._client: AsyncOpenAI | None = None
        # Whether to ask for streamed token usage. Disabled permanently after
        # a provider rejects the stream_options parameter.
        self._include_usage = True

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = resolve_api_key(self.spec, self.config)
            if not api_key:
                env = self.spec.provider.api_key_env
                raise ValueError(
                    f"No API key for provider {self.spec.provider.id!r}. "
                    f"Set [llm] api_key in ~/.limbo/config.toml"
                    + (f" or the ${env} environment variable." if env else ".")
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=resolve_base_url(self.spec, self.config),
                default_headers=resolve_headers(self.spec, self.config) or None,
                # SDK built-in retries are disabled: stream_with_retry owns
                # retrying so every attempt is trace-visible via on_request.
                max_retries=0,
                timeout=httpx.Timeout(
                    self.config.llm.timeout,
                    connect=self.config.llm.connect_timeout,
                ),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if it has been created."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _thinking_params(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build thinking-control parameters for the model's dialect.

        Returns ``(params, extra_body)``: SDK-typed parameters (e.g.
        ``reasoning_effort``) versus raw body fields the SDK does not model
        (``thinking`` for the deepseek/zai dialects).
        """
        spec = self.spec
        if not spec.reasoning or spec.thinking_format is None:
            return {}, {}
        effort = self.config.llm.thinking_effort
        if effort is None:
            return {}, {}
        if spec.thinking_format == "openai":
            value = spec.thinking_levels.get(effort)
            if value is None:
                warnings.warn(
                    f"thinking_effort={effort!r} is not supported by {spec.id} "
                    f"(supported: {sorted(spec.thinking_levels)}); ignoring.",
                    stacklevel=2,
                )
                return {}, {}
            return {"reasoning_effort": value}, {}
        if spec.thinking_format in ("deepseek", "zai"):
            if effort == "off":
                if not spec.thinking_can_disable:
                    warnings.warn(
                        f"{spec.id} does not support disabling thinking; ignoring.",
                        stacklevel=2,
                    )
                    return {}, {}
                return {}, {"thinking": {"type": "disabled"}}
            thinking: dict[str, Any] = {"type": "enabled"}
            if spec.thinking_format == "zai":
                # z.ai/GLM dialect: clear_thinking:false keeps thinking across
                # turns (pi's zai thinkingFormat).
                thinking["clear_thinking"] = False
            params: dict[str, Any] = {}
            if spec.thinking_format == "zai" and spec.thinking_levels:
                # glm-5.2 additionally takes reasoning_effort (pi sends both).
                value = spec.thinking_levels.get(effort)
                if value is None:
                    warnings.warn(
                        f"thinking_effort={effort!r} is not supported by {spec.id} "
                        f"(supported: {sorted(spec.thinking_levels)}); ignoring.",
                        stacklevel=2,
                    )
                else:
                    params["reasoning_effort"] = value
            return params, {"thinking": thinking}
        return {}, {}

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
        include_reasoning = self.spec.requires_reasoning_content
        request_messages = [
            _message_to_openai(m, include_reasoning=include_reasoning)
            for m in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": request_messages,
            "temperature": self.config.llm.temperature,
            "max_tokens": self.config.llm.max_tokens or self.spec.max_tokens,
            "stream": True,
        }
        thinking_params, thinking_extra = self._thinking_params()
        kwargs.update(thinking_params)
        # Provider extra-body fields (quirks the SDK doesn't type); thinking
        # extras win on conflict, tool-scoped extras apply with tools only.
        extra_body: dict[str, Any] = dict(self.spec.provider.extra_body)
        extra_body.update(thinking_extra)
        if tools:
            kwargs["tools"] = tools
            extra_body.update(self.spec.provider.tool_extra_body)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if self._include_usage:
            kwargs["stream_options"] = {"include_usage": True}

        start = time.monotonic()
        try:
            stream = await self._create(kwargs, on_request)
        except BadRequestError as e:
            # Some providers reject stream_options; retry once without it and
            # stop asking for usage on later requests.
            if not self._include_usage or not _is_stream_options_error(e):
                raise
            self._include_usage = False
            kwargs.pop("stream_options", None)
            stream = await self._create(kwargs, on_request)

        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        ttft: float | None = None
        active_calls: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start
            if chunk.usage is not None:
                # The final chunk carries usage when include_usage is on.
                # model_dump keeps provider extras such as DeepSeek's
                # prompt_cache_hit_tokens / prompt_cache_miss_tokens.
                try:
                    usage = chunk.usage.model_dump()
                except Exception:  # noqa: BLE001
                    usage = {"raw": str(chunk.usage)}
            if not chunk.choices:
                continue
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            delta = chunk.choices[0].delta
            # Reasoning models stream thinking in a separate field; the name
            # varies by provider (Moonshot/DeepSeek: reasoning_content).
            reasoning = None
            for attr in ("reasoning_content", "reasoning"):
                value = getattr(delta, attr, None)
                if isinstance(value, str) and value:
                    reasoning = value
                    break
            if reasoning:
                yield ThinkingChunk(text=reasoning)
            if delta.content:
                yield TextChunk(text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_calls:
                        active_calls[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    current = active_calls[idx]
                    if tc.id:
                        current["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            current["name"] += tc.function.name
                        if tc.function.arguments:
                            current["arguments"] += tc.function.arguments

        for idx in sorted(active_calls):
            call = active_calls[idx]
            raw_arguments = call["arguments"]
            try:
                args = json.loads(raw_arguments) if raw_arguments else {}
            except json.JSONDecodeError as e:
                args = {
                    "raw_arguments": raw_arguments,
                    "parse_error": str(e),
                }
            yield ToolCallEvent(
                id=call["id"] or f"call_{idx}",
                name=call["name"],
                arguments=args,
            )

        yield CompletionMeta(
            usage=usage,
            finish_reason=finish_reason,
            ttft=ttft,
            duration=time.monotonic() - start,
        )

    async def _create(
        self, kwargs: dict[str, Any], on_request: RequestHook | None
    ):
        """Fire one HTTP attempt, reporting the exact body via the hook."""
        if on_request is not None:
            on_request(kwargs)
        return await self.client.chat.completions.create(**kwargs)


def _is_stream_options_error(error: BadRequestError) -> bool:
    message = str(error).lower()
    return "stream_options" in message or "include_usage" in message


def _message_to_openai(
    message: Message, *, include_reasoning: bool = False
) -> dict[str, Any]:
    m: dict[str, Any] = {"role": message.role}
    # OpenAI requires `content` on assistant and tool messages, even when empty.
    m["content"] = message.content or ""
    if message.role == "user" and message.attachments:
        # Multimodal user message: image_url blocks + text. Missing files
        # (expired session attachments) are skipped; if none survive, the
        # message stays a plain string.
        blocks: list[dict[str, Any]] = []
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
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )
        if blocks:
            blocks.append({"type": "text", "text": message.content or ""})
            m["content"] = blocks
    if include_reasoning and message.role == "assistant":
        # Kimi K3 requires reasoning_content on replayed assistant messages;
        # send the stored thinking or an empty string.
        m["reasoning_content"] = message.reasoning or ""
    if message.tool_calls:
        # The OpenAI SDK requires function.arguments to be a JSON string.
        # Internal messages keep arguments as a dict for validation; serialize
        # only when building the API request.
        m["tool_calls"] = [
            {
                **tc,
                "function": {
                    **tc["function"],
                    "arguments": (
                        json.dumps(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], dict)
                        else tc["function"]["arguments"]
                    ),
                },
            }
            for tc in message.tool_calls
        ]
    if message.tool_call_id:
        m["tool_call_id"] = message.tool_call_id
    return m
