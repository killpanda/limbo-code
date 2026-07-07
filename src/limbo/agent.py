"""Agent conversation loop."""

from __future__ import annotations

import asyncio
import os
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.models import Message, TextChunk, ToolCallEvent, ToolResult
from limbo.tools.registry import ToolRegistry


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultEvent:
    id: str
    name: str
    result: ToolResult
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ErrorEvent:
    message: str


AgentEvent = TextDelta | ToolCallRequest | ToolResultEvent | ErrorEvent


class Agent:
    """Orchestrates the conversation between user, LLM, and tools."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        workdir: Path,
        session_dir: Path | None = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.workdir = workdir
        self.registry = ToolRegistry(workdir=workdir, config=config)
        self.messages: list[Message] = []
        self._pending_tool: dict[str, Any] | None = None
        self._confirmation_applied = False
        self._iteration_count = 0
        self._pending_placeholders: set[str] = set()
        self._last_saved_index: int = 0
        self._init_system_message()

        self._session_dir = session_dir or Path.home() / ".limbo" / "sessions"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        self._session_file = self._session_dir / f"{timestamp}-{os.getpid()}.jsonl"

    @property
    def confirmation_applied(self) -> bool:
        """Whether a confirmed tool has been applied and the loop can resume."""
        return self._confirmation_applied

    def _init_system_message(self) -> None:
        self.messages.append(
            Message(
                role="system",
                content=(
                    "You are an expert coding assistant operating inside limbo, "
                    "a coding agent harness. You help users by reading files, "
                    "executing commands, editing code, and writing new files.\n\n"
                    "Available tools:\n"
                    "- read: Read file contents\n"
                    "- bash: Execute bash commands\n"
                    "- edit: Make surgical edits to files (find exact text and replace)\n"
                    "- write: Create or overwrite files\n"
                    "- grep: Search file contents for patterns (respects root .gitignore)\n"
                    "- find: Find files by glob pattern (respects root .gitignore only)\n"
                    "- ls: List directory contents\n\n"
                    "Guidelines:\n"
                    "- Prefer grep/find/ls tools over bash for file exploration\n"
                    "- Use read to examine files before editing\n"
                    "- Use edit for precise changes (old_text must match exactly)\n"
                    "- Use write only for new files or complete rewrites\n"
                    "- Be concise in your responses\n"
                    "- Show file paths clearly when working with files"
                ),
            )
        )

    def reject_pending_tool(self, reason: str = "Action rejected.") -> None:
        """Clear any tool awaiting confirmation.

        Updates the placeholder ``role="tool"`` message installed when the
        pending call was first encountered, or appends one if it is missing,
        so the assistant message that referenced the tool call remains valid
        for the OpenAI API. Any later tool calls in the same assistant turn
        are marked as not executed.
        """
        if self._pending_tool is None:
            return
        pending_id = self._pending_tool["id"]

        # Find the most recent assistant message that owns this pending call
        # so we can update its later tool-call placeholders as well.
        assistant = None
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.tool_calls:
                if any(tc["id"] == pending_id for tc in msg.tool_calls):
                    assistant = msg
                    break

        if assistant is not None and assistant.tool_calls:
            pending_idx: int | None = None
            for idx, tc in enumerate(assistant.tool_calls):
                if tc["id"] == pending_id:
                    pending_idx = idx
                    break
            if pending_idx is not None:
                for tc in assistant.tool_calls[pending_idx:]:
                    content = (
                        reason
                        if tc["id"] == pending_id
                        else "Action not executed: earlier tool was rejected."
                    )
                    for idx, existing in enumerate(self.messages):
                        if existing.role == "tool" and existing.tool_call_id == tc["id"]:
                            self.messages[idx] = existing.model_copy(
                                update={"content": content}
                            )
                            break
                    else:
                        self.messages.append(
                            Message(
                                role="tool",
                                content=content,
                                tool_call_id=tc["id"],
                            )
                        )
                    self._pending_placeholders.discard(tc["id"])
                self._pending_tool = None
                return

        # Fallback when the owning assistant message cannot be located.
        for idx, msg in enumerate(self.messages):
            if msg.role == "tool" and msg.tool_call_id == pending_id:
                self.messages[idx] = msg.model_copy(update={"content": reason})
                break
        else:
            self.messages.append(
                Message(
                    role="tool",
                    content=reason,
                    tool_call_id=pending_id,
                )
            )
        self._pending_placeholders.discard(pending_id)
        self._pending_tool = None

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        # Reset per-user-turn state and discard stale pending confirmations.
        self._iteration_count = 0
        self._confirmation_applied = False
        if self._pending_tool is not None:
            self.reject_pending_tool(
                reason="Action timed out or was superseded by a new turn."
            )

        self.messages.append(Message(role="user", content=user_input))

        try:
            async for event in self._conversation_loop():
                yield event
        finally:
            try:
                await self._save_session()
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Failed to save session: {e}", stacklevel=2)

    async def continue_after_confirmation(self) -> AsyncIterator[AgentEvent]:
        """Resume the conversation loop after a confirmed tool was applied.

        Before returning to the LLM, any remaining tool calls from the
        original assistant turn are executed and their placeholder results are
        replaced with real results. If a remaining tool requires confirmation,
        the loop stops and yields its confirmation event.

        Yields the remaining assistant responses and any follow-up tool events
        until the assistant produces a final response without pending tool calls.
        """
        if self._pending_tool is not None:
            yield ErrorEvent(
                message="Cannot continue: a tool is still pending confirmation."
            )
            return
        if not self._confirmation_applied:
            yield ErrorEvent(message="Cannot continue: no confirmed tool to resume from.")
            return
        self._confirmation_applied = False

        try:
            async for event in self._execute_remaining_tools():
                yield event
            async for event in self._conversation_loop():
                yield event
        finally:
            try:
                await self._save_session()
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Failed to save session: {e}", stacklevel=2)

    async def _execute_remaining_tools(self) -> AsyncIterator[AgentEvent]:
        """Execute remaining placeholder tool calls from the last assistant turn.

        The assistant message already references every tool call in a multi-tool
        turn, but the calls after a confirmation pause only have placeholder
        results. Run those calls, replacing placeholders with real outputs, and
        stop if another call requires confirmation.
        """
        assistant = None
        for msg in reversed(self.messages):
            if msg.role == "assistant" and msg.tool_calls:
                assistant = msg
                break
        if assistant is None or assistant.tool_calls is None:
            return

        for tc in assistant.tool_calls:
            if tc["id"] not in self._pending_placeholders:
                continue

            name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
            try:
                result = await self.registry.execute(name, arguments, dry_run=True)
            except Exception as e:  # noqa: BLE001
                yield ErrorEvent(message=f"Tool error: {e}")
                return
            yield ToolResultEvent(
                id=tc["id"], name=name, result=result, arguments=arguments
            )
            if result.requires_confirmation:
                self._pending_tool = tc
                return
            for idx, msg in enumerate(self.messages):
                if msg.role == "tool" and msg.tool_call_id == tc["id"]:
                    self.messages[idx] = msg.model_copy(
                        update={"content": result.output or result.error or ""}
                    )
                    break
            self._pending_placeholders.discard(tc["id"])

    async def _conversation_loop(self) -> AsyncIterator[AgentEvent]:
        while self._iteration_count < self.config.llm.max_iterations:
            self._iteration_count += 1
            try:
                async for event in self._call_llm():
                    yield event
            except Exception as e:  # noqa: BLE001
                yield ErrorEvent(message=f"LLM error: {e}")
                return

            last = self.messages[-1]
            if not last.tool_calls:
                break

            for idx, tc in enumerate(last.tool_calls):
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
                try:
                    result = await self.registry.execute(name, arguments, dry_run=True)
                except Exception as e:  # noqa: BLE001
                    yield ErrorEvent(message=f"Tool error: {e}")
                    return
                if result.requires_confirmation:
                    self._pending_tool = tc
                    # The assistant message already references every tool call in
                    # this turn, but only the calls before this one have results.
                    # Append placeholders for this pending call and the remaining
                    # calls so the message history stays valid for OpenAI. The
                    # pending call's placeholder is replaced by the real result
                    # when the user confirms it. Placeholders are installed
                    # before yielding the ToolResultEvent so confirmation handlers
                    # can update them immediately.
                    for pending_or_remaining in last.tool_calls[idx:]:
                        self.messages.append(
                            Message(
                                role="tool",
                                content="Action pending user confirmation.",
                                tool_call_id=pending_or_remaining["id"],
                            )
                        )
                        self._pending_placeholders.add(pending_or_remaining["id"])
                yield ToolResultEvent(
                    id=tc["id"], name=name, result=result, arguments=arguments
                )
                if result.requires_confirmation:
                    return
                self.messages.append(
                    Message(
                        role="tool",
                        content=result.output or result.error or "",
                        tool_call_id=tc["id"],
                    )
                )
                self._pending_placeholders.discard(tc["id"])

    async def apply_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Apply the tool that is currently pending confirmation.

        Validates that ``name`` and ``arguments`` match the stored pending tool
        call. Returns an error result and leaves the pending state untouched
        when they do not match.
        """
        if self._pending_tool is None:
            return ToolResult(success=False, error="No tool is pending confirmation.")

        pending_name = self._pending_tool["function"]["name"]
        pending_arguments = self._pending_tool["function"]["arguments"]
        if name != pending_name or arguments != pending_arguments:
            return ToolResult(
                success=False,
                error=(
                    f"Tool {name} does not match pending tool {pending_name}."
                ),
            )

        result = await self.registry.execute(name, arguments, dry_run=False)
        # Replace the placeholder tool result installed when the pending call
        # was first encountered, so the message history stays valid and shows
        # the actual outcome.
        for idx, msg in enumerate(self.messages):
            if (
                msg.role == "tool"
                and msg.tool_call_id == self._pending_tool["id"]
            ):
                self.messages[idx] = msg.model_copy(
                    update={"content": result.output or result.error or ""}
                )
                break
        else:
            self.messages.append(
                Message(
                    role="tool",
                    content=result.output or result.error or "",
                    tool_call_id=self._pending_tool["id"],
                )
            )
        self._pending_placeholders.discard(self._pending_tool["id"])
        self._pending_tool = None
        self._confirmation_applied = True
        return result

    async def _call_llm(self) -> AsyncIterator[AgentEvent]:
        tool_definitions = self.registry.definitions()
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []

        async for event in self.llm_client.chat(self.messages, tools=tool_definitions):
            if isinstance(event, TextChunk):
                assistant_content += event.text
                yield TextDelta(text=event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(
                    {
                        "id": event.id,
                        "type": "function",
                        "function": {
                            "name": event.name,
                            "arguments": event.arguments,
                        },
                    }
                )
                yield ToolCallRequest(
                    id=event.id, name=event.name, arguments=event.arguments
                )

        self.messages.append(
            Message(
                role="assistant",
                content=assistant_content or None,
                tool_calls=tool_calls if tool_calls else None,
            )
        )

    def _save_session_sync(self) -> None:
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._session_file.exists() and self._last_saved_index > 0 else "w"
        with self._session_file.open(mode, encoding="utf-8") as f:
            for msg in self.messages[self._last_saved_index :]:
                f.write(msg.model_dump_json() + "\n")
        self._last_saved_index = len(self.messages)

    async def _save_session(self) -> None:
        await asyncio.to_thread(self._save_session_sync)
