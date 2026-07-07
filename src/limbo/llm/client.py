"""LLM client protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from limbo.models import LLMEvent, Message


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMEvent]: ...
