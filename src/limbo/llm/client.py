"""LLM client protocol."""

from __future__ import annotations

from typing import Any, Iterator, Protocol

from limbo.models import LLMEvent, Message


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Iterator[LLMEvent]: ...
