"""LLM client protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from limbo.models import LLMEvent, Message

# Called with the exact request body (or SDK kwargs) about to be sent, on
# every HTTP attempt. Used for trace logging; must not mutate the body.
RequestHook = Callable[[dict[str, Any]], None]

# Dialect prefix for ``Message.reasoning_signature`` values stored by the
# Responses client. Signatures are dialect-specific: a value carrying this
# prefix is a serialized Responses reasoning item, and other dialects must
# skip it on replay instead of sending it as their own signature shape.
RESPONSES_SIGNATURE_PREFIX = "responses:"


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_request: RequestHook | None = None,
    ) -> AsyncIterator[LLMEvent]: ...
