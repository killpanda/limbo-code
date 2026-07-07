"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
from typing import Any, Iterator

from openai import OpenAI

from limbo.config import Config
from limbo.models import LLMEvent, Message, TextChunk, ToolCallEvent


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat completions with streaming."""

    def __init__(self, config: Config):
        self.config = config
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.llm.api_key or "",
                base_url=self.config.llm.base_url,
            )
        return self._client

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Iterator[LLMEvent]:
        request_messages = [_message_to_openai(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": request_messages,
            "temperature": self.config.llm.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = self.client.chat.completions.create(**kwargs)

        active_calls: dict[int, dict[str, Any]] = {}
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield TextChunk(text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_calls:
                        active_calls[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
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
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
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
        m["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        m["tool_call_id"] = message.tool_call_id
    return m
