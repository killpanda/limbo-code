"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from limbo.config import Config
from limbo.llm.catalog import (
    ModelSpec,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from limbo.models import LLMEvent, Message, TextChunk, ThinkingChunk, ToolCallEvent


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

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = resolve_api_key(self.spec, self.config.llm.api_key)
            if not api_key:
                env = self.spec.provider.api_key_env
                raise ValueError(
                    f"No API key for provider {self.spec.provider.id!r}. "
                    f"Set [llm] api_key in ~/.limbo/config.toml"
                    + (f" or the ${env} environment variable." if env else ".")
                )
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=resolve_base_url(self.spec, self.config.llm.base_url),
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if it has been created."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _thinking_params(self) -> dict[str, Any]:
        """Build thinking-control parameters for the model's dialect."""
        spec = self.spec
        if not spec.reasoning or spec.thinking_format is None:
            return {}
        effort = self.config.llm.thinking_effort
        if effort is None:
            return {}
        if spec.thinking_format == "openai":
            value = spec.thinking_levels.get(effort)
            if value is None:
                warnings.warn(
                    f"thinking_effort={effort!r} is not supported by {spec.id} "
                    f"(supported: {sorted(spec.thinking_levels)}); ignoring.",
                    stacklevel=2,
                )
                return {}
            return {"reasoning_effort": value}
        if spec.thinking_format == "deepseek":
            if effort == "off":
                if not spec.thinking_can_disable:
                    warnings.warn(
                        f"{spec.id} does not support disabling thinking; ignoring.",
                        stacklevel=2,
                    )
                    return {}
                return {"thinking": {"type": "disabled"}}
            return {"thinking": {"type": "enabled"}}
        return {}

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMEvent]:
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
        thinking = self._thinking_params()
        if self.spec.thinking_format == "deepseek" and thinking:
            # `thinking` is not an OpenAI SDK parameter; pass it through the
            # request body untouched (Kimi K2 thinking dialect).
            kwargs["extra_body"] = thinking
        else:
            kwargs.update(thinking)
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        active_calls: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
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


def _message_to_openai(
    message: Message, *, include_reasoning: bool = False
) -> dict[str, Any]:
    m: dict[str, Any] = {"role": message.role}
    # OpenAI requires `content` on assistant and tool messages, even when empty.
    m["content"] = message.content or ""
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
