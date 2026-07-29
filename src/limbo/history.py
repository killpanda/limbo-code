"""Message-history bookkeeping for tool calls.

Owns the invariant that keeps the OpenAI API happy: every assistant
``tool_call`` must have exactly one ``role="tool"`` result message. All
find-and-replace logic for placeholder results lives in this module instead
of being scattered across the agent loop.
"""

from __future__ import annotations

from limbo.models import Message

PENDING_CONTENT = "Action pending user confirmation."
INTERRUPTED_CONTENT = "[session restored: tool call interrupted]"
NOT_EXECUTED_AFTER_REJECTION = "Action not executed: earlier tool was rejected."
NOT_EXECUTED_AFTER_FAILURE = "Action not executed: earlier tool failed."


class ToolHistory:
    """Bookkeeper for tool-call results inside a message list."""

    def __init__(
        self, messages: list[Message], pending_ids: set[str] | None = None
    ):
        self.messages = messages
        self.pending_ids: set[str] = (
            pending_ids if pending_ids is not None else set()
        )

    def latest_assistant_with_tools(self) -> Message | None:
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.tool_calls:
                return msg
        return None

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
        self.pending_ids.discard(tool_call_id)

    def install_placeholders(self, tool_call_ids: list[str]) -> None:
        """Install pending-result placeholders and track them."""
        for tool_call_id in tool_call_ids:
            self.messages.append(
                Message(
                    role="tool",
                    content=PENDING_CONTENT,
                    tool_call_id=tool_call_id,
                )
            )
            self.pending_ids.add(tool_call_id)

    def record_error(
        self,
        assistant: Message,
        start_idx: int,
        crashed_id: str,
        error_message: str,
    ) -> None:
        """Record an error for a crashed call and cancel its later siblings."""
        tool_calls = assistant.tool_calls or []
        for idx in range(start_idx, len(tool_calls)):
            tc = tool_calls[idx]
            content = (
                error_message
                if tc["id"] == crashed_id
                else NOT_EXECUTED_AFTER_FAILURE
            )
            self.record_result(tc["id"], content)

    def record_rejection(
        self, pending_id: str, reason: str = "Action rejected."
    ) -> None:
        """Record a rejection for a pending call and cancel its later siblings."""
        # Find the most recent assistant message that owns this pending call.
        assistant = None
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.tool_calls:
                if any(tc["id"] == pending_id for tc in msg.tool_calls):
                    assistant = msg
                    break

        if assistant is not None and assistant.tool_calls:
            pending_idx = next(
                (
                    idx
                    for idx, tc in enumerate(assistant.tool_calls)
                    if tc["id"] == pending_id
                ),
                None,
            )
            if pending_idx is not None:
                for tc in assistant.tool_calls[pending_idx:]:
                    content = (
                        reason
                        if tc["id"] == pending_id
                        else NOT_EXECUTED_AFTER_REJECTION
                    )
                    self.record_result(tc["id"], content)
                return

        # Fallback when the owning assistant message cannot be located.
        self.record_result(pending_id, reason)


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
