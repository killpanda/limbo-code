"""Message-history bookkeeping for tool calls.

Owns the invariant that keeps the OpenAI API happy: every assistant
``tool_call`` must have exactly one ``role="tool"`` result message.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from limbo.models import Attachment, Message

# Placeholder for a dangling call whose fate is unknown: a legacy session
# (no intent records), or a read-only call queued for re-run on resume.
INTERRUPTED_CONTENT = "[session restored: tool call interrupted]"
# The call's intent was never committed, so the tool never started.
NOT_STARTED_CONTENT = (
    "[session restored: tool call interrupted before it started; "
    "it did not run]"
)
# The intent was committed and the tool has side effects: it may have
# done part of its work before the process died.
PARTIAL_CONTENT = (
    "[session restored: tool call interrupted while running; it may have "
    "partially run. Check its effects before retrying.]"
)


class ToolHistory:
    """Bookkeeper for tool-call results inside a message list."""

    def __init__(self, messages: list[Message]):
        self.messages = messages

    def record_result(
        self,
        tool_call_id: str,
        content: str,
        attachments: list[Attachment] | None = None,
    ) -> None:
        """Replace the existing result for a call, or append a new one.

        ``attachments`` carries image payloads from tools (e.g. read on an
        image file); they are stored on the tool message and replayed as
        multimodal blocks by the clients, same as user-message attachments.
        """
        for idx, msg in enumerate(self.messages):
            if msg.role == "tool" and msg.tool_call_id == tool_call_id:
                self.messages[idx] = msg.model_copy(
                    update={"content": content, "attachments": attachments}
                )
                break
        else:
            self.messages.append(
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=tool_call_id,
                    attachments=attachments,
                )
            )


@dataclass
class Recovery:
    """Repaired history plus the read-only calls to re-run on resume."""

    messages: list[Message]
    # Tool-call dicts (as stored on the assistant message) whose placeholder
    # result should be replaced by a real re-run.
    rerun: list[dict[str, Any]] = field(default_factory=list)


def recover(
    messages: list[Message],
    *,
    started: Collection[str] | None = None,
    is_safe: Callable[[str], bool] = lambda _name: False,
) -> Recovery:
    """Prepare persisted messages for reuse as LLM history.

    - Drops the leading system message (a fresh one is generated on resume).
    - Repairs dangling tool_calls: if the previous session crashed between a
      tool call and its result, placeholder tool messages are inserted
      immediately after the assistant message so the API history stays valid.

    The placeholder says what is known about the call (pi-durable's effect
    sandwich): ``started`` holds the ids whose intent record was committed
    before the tool ran (``None`` for a legacy session that never wrote
    intents). Calls to tools ``is_safe`` approves are queued in
    ``Recovery.rerun`` instead — re-running a read-only call is harmless.
    """
    history = list(messages)
    if history and history[0].role == "system":
        history = history[1:]

    restored: list[Message] = []
    rerun: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def placeholder(tc: dict[str, Any]) -> str:
        if is_safe(tc.get("function", {}).get("name", "")):
            rerun.append(tc)
            return INTERRUPTED_CONTENT
        if started is None:
            return INTERRUPTED_CONTENT
        return PARTIAL_CONTENT if tc["id"] in started else NOT_STARTED_CONTENT

    def flush_pending() -> None:
        for tc in pending:
            restored.append(
                Message(
                    role="tool",
                    tool_call_id=tc["id"],
                    content=placeholder(tc),
                )
            )
        pending.clear()

    for msg in history:
        if msg.role == "assistant" and msg.tool_calls:
            flush_pending()
            pending.extend(tc for tc in msg.tool_calls if tc.get("id"))
        elif msg.role == "tool":
            pending[:] = [tc for tc in pending if tc["id"] != msg.tool_call_id]
        else:
            flush_pending()
        restored.append(msg)
    flush_pending()
    return Recovery(messages=restored, rerun=rerun)


def repair(messages: list[Message]) -> list[Message]:
    """:func:`recover` without intent information (legacy placeholders)."""
    return recover(messages).messages
