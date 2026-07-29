"""LLM client protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from limbo.models import LLMEvent, Message

# Called with the exact request body (or SDK kwargs) about to be sent, on
# every HTTP attempt. Used for trace logging; must not mutate the body.
RequestHook = Callable[[dict[str, Any]], None]


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_request: RequestHook | None = None,
    ) -> AsyncIterator[LLMEvent]: ...
