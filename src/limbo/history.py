"""Message-history bookkeeping for tool calls.

Owns the invariant that keeps the OpenAI API happy: every assistant
``tool_call`` must have exactly one ``role="tool"`` result message.
"""

from __future__ import annotations

from limbo.models import Message

INTERRUPTED_CONTENT = "[session restored: tool call interrupted]"


class ToolHistory:
    """Bookkeeper for tool-call results inside a message list."""

    def __init__(self, messages: list[Message]):
        self.messages = messages

    def record_result(self, tool_call_id: str, content: str) -> None:
        """Replace the existing result for a call, or append a new one."""
        for idx, msg in enumerate(self.messages):
            if msg.role == "tool" and msg.tool_call_id == tool_call_id:
                self.messages[idx] = msg.model_copy(update={"content": content})
                break
        else:
            self.messages.append(
                Message(role="tool", content=content, tool_call_id=tool_call_id)
            )


def repair(messages: list[Message]) -> list[Message]:
    """Prepare persisted messages for reuse as LLM history.

    - Drops the leading system message (a fresh one is generated on resume).
    - Repairs dangling tool_calls: if the previous session crashed between a
      tool call and its result, placeholder tool messages are inserted
      immediately after the assistant message so the API history stays valid.
    """
    history = list(messages)
    if history and history[0].role == "system":
        history = history[1:]

    restored: list[Message] = []
    pending_tool_ids: list[str] = []

    def flush_pending() -> None:
        for tool_id in pending_tool_ids:
            restored.append(
                Message(
                    role="tool",
                    tool_call_id=tool_id,
                    content=INTERRUPTED_CONTENT,
                )
            )
        pending_tool_ids.clear()

    for msg in history:
        if msg.role == "assistant" and msg.tool_calls:
            flush_pending()
            pending_tool_ids.extend(
                tc["id"] for tc in msg.tool_calls if tc.get("id")
            )
        elif msg.role == "tool":
            if msg.tool_call_id in pending_tool_ids:
                pending_tool_ids.remove(msg.tool_call_id)
        else:
            flush_pending()
        restored.append(msg)
    flush_pending()
    return restored
