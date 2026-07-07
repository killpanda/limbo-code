"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from limbo.config import Config
from limbo.models import LLMEvent, Message, TextChunk, ToolCallEvent


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat completions with streaming."""

    def __init__(self, config: Config):
        self.config = config
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.llm.api_key or "",
                base_url=self.config.llm.base_url,
            )
        return self._client

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMEvent]:
        request_messages = [_message_to_openai(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": request_messages,
            "temperature": self.config.llm.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self.client.chat.completions.create(**kwargs)

        active_calls: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
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


def _message_to_openai(message: Message) -> dict[str, Any]:
    m: dict[str, Any] = {"role": message.role}
    if message.content:
        m["content"] = message.content
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
