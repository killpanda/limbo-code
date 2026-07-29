"""Agent conversation loop."""

from __future__ import annotations

import asyncio
import os
import platform
import secrets
import time
import traceback
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from limbo import __version__
from limbo.config import Config
from limbo.history import ToolHistory
from limbo.history import repair as repair_history
from limbo.llm.client import LLMClient
from limbo.models import (
    CompletionMeta,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
    ToolResult,
)
from limbo.sessions import SessionMeta, derive_title, load_session, save_session
from limbo.tools.registry import ToolRegistry
from limbo.trace import TraceLogger, trace_path_for


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ThinkingDelta:
    """A streamed reasoning/thinking delta (reasoning models only)."""

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


AgentEvent = TextDelta | ThinkingDelta | ToolCallRequest | ToolResultEvent | ErrorEvent


def _extract_cached_tokens(usage: dict[str, Any] | None) -> int | None:
    """Normalize provider-specific prompt cache-hit counters.

    DeepSeek reports ``prompt_cache_hit_tokens``, OpenAI nests
    ``prompt_tokens_details.cached_tokens``, Anthropic reports
    ``cache_read_input_tokens``.
    """
    if not usage:
        return None
    hit = usage.get("prompt_cache_hit_tokens")
    if isinstance(hit, int):
        return hit
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            return cached
    read = usage.get("cache_read_input_tokens")
    return read if isinstance(read, int) else None


class Agent:
    """Orchestrates the conversation between user, LLM, and tools."""

    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        workdir: Path,
        session_dir: Path | None = None,
        resume: Path | None = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.workdir = workdir
        self.registry = ToolRegistry(workdir=workdir, config=config)
        self._history = ToolHistory([])
        self._pending_tool: dict[str, Any] | None = None
        self._confirmation_applied = False
        self._iteration_count = 0
        self._init_system_message()

        self._session_dir = session_dir or Path.home() / ".limbo" / "sessions"
        if resume is not None:
            self._meta, history = load_session(resume)
            self._session_file = resume
            self.messages.extend(repair_history(history))
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            # Random suffix avoids collisions when sessions are created in
            # quick succession within the same process (e.g. `/new`).
            self._session_file = (
                self._session_dir
                / f"{timestamp}-{os.getpid()}-{secrets.token_hex(2)}.jsonl"
            )
            self._meta = SessionMeta(
                id=self._session_file.stem,
                workdir=str(workdir.resolve()),
                model=config.llm.model,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )

        # Trace log: full-fidelity JSONL record of the run, kept separate from
        # the resumable session file. Created lazily-tolerant: tracing must
        # never break the agent.
        self.trace = TraceLogger(trace_path_for(self._session_file))
        self._turn_count = 0
        self._turn_start: float | None = None
        self._pending_since: float | None = None
        self.trace.log(
            "session_start",
            session_id=self._meta.id,
            workdir=str(workdir.resolve()),
            resumed=resume is not None,
            restored_messages=(len(self.messages) - 1) if resume is not None else 0,
            limbo_version=__version__,
            python=platform.python_version(),
            platform=platform.platform(),
            config={
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                "max_iterations": config.llm.max_iterations,
                "thinking_effort": config.llm.thinking_effort,
                "bash_enabled": config.tools.bash_enabled,
            },
        )

    @property
    def session_id(self) -> str:
        """Stable id of the current session (the session file stem)."""
        return self._meta.id

    @property
    def session_meta(self) -> SessionMeta:
        """Metadata of the current session (title filled in on first save)."""
        if not self._meta.title:
            self._meta.title = derive_title(self.messages)
        return self._meta

    @property
    def confirmation_applied(self) -> bool:
        """Whether a confirmed tool has been applied and the loop can resume."""
        return self._confirmation_applied

    def _init_system_message(self) -> None:
        content_parts = [
            (
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
        ]
        # Load optional project-level AGENTS.md for extra context.
        project_md = self.workdir / "AGENTS.md"
        if project_md.is_file():
            try:
                text = project_md.read_text(encoding="utf-8")
                content_parts.append(
                    f"\n\n## Project context from AGENTS.md\n\n{text}"
                )
            except OSError:
                pass
        # Load optional global AGENTS.md for personal preferences.
        global_md = Path.home() / ".limbo" / "AGENTS.md"
        if global_md.is_file():
            try:
                text = global_md.read_text(encoding="utf-8")
                content_parts.append(
                    f"\n\n## User preferences from ~/.limbo/AGENTS.md\n\n{text}"
                )
            except OSError:
                pass
        self.messages.append(
            Message(
                role="system",
                content="".join(content_parts),
            )
        )

    @property
    def messages(self) -> list[Message]:
        """The conversation history (owned by the tool-call bookkeeper)."""
        return self._history.messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._history.messages = value

    @property
    def _pending_placeholders(self) -> set[str]:
        """Backward-compatible view of the ledger's pending placeholder ids."""
        return self._history.pending_ids

    def reject_pending_tool(
        self, reason: str = "Action rejected.", decision: str = "rejected"
    ) -> None:
        """Clear any tool awaiting confirmation.

        Records the rejection in the message history (cancelling any later
        tool calls from the same assistant turn) so the assistant message
        that referenced the tool call remains valid for the OpenAI API.
        """
        if self._pending_tool is None:
            return
        self._history.record_rejection(self._pending_tool["id"], reason)
        self.trace.log(
            "confirmation",
            turn=self._turn_count,
            decision=decision,
            reason=reason,
            tool_call_id=self._pending_tool["id"],
            name=self._pending_tool["function"]["name"],
            wait=self._pending_wait(),
        )
        self._pending_tool = None
        self._pending_since = None

    def _pending_wait(self) -> float | None:
        """Seconds the current confirmation has been waiting, if known."""
        if self._pending_since is None:
            return None
        return time.monotonic() - self._pending_since

    def _log_turn_end(self) -> None:
        if self._turn_start is None:
            return
        self.trace.log(
            "turn_end",
            turn=self._turn_count,
            duration=time.monotonic() - self._turn_start,
            iterations=self._iteration_count,
            status="awaiting_confirmation" if self._pending_tool else "completed",
        )
        self._turn_start = None

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        # Reset per-user-turn state and discard stale pending confirmations.
        self._iteration_count = 0
        self._confirmation_applied = False
        self._turn_count += 1
        self._turn_start = time.monotonic()
        if self._pending_tool is not None:
            self.reject_pending_tool(
                reason="Action timed out or was superseded by a new turn.",
                decision="superseded",
            )

        self.messages.append(Message(role="user", content=user_input))
        self.trace.log(
            "user_message", turn=self._turn_count, content=user_input
        )

        try:
            async for event in self._conversation_loop():
                yield event
        finally:
            self._log_turn_end()
            try:
                await self._save_session()
            except Exception as e:  # noqa: BLE001
                self.trace.log(
                    "session_save_error", turn=self._turn_count, error=str(e)
                )
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
            self.trace.log(
                "error",
                turn=self._turn_count,
                kind="invalid_continuation",
                message="Cannot continue: a tool is still pending confirmation.",
            )
            yield ErrorEvent(
                message="Cannot continue: a tool is still pending confirmation."
            )
            return
        if not self._confirmation_applied:
            self.trace.log(
                "error",
                turn=self._turn_count,
                kind="invalid_continuation",
                message="Cannot continue: no confirmed tool to resume from.",
            )
            yield ErrorEvent(message="Cannot continue: no confirmed tool to resume from.")
            return
        self._confirmation_applied = False
        self._turn_start = time.monotonic()

        try:
            async for event in self._execute_remaining_tools():
                yield event
            async for event in self._conversation_loop():
                yield event
        finally:
            self._log_turn_end()
            try:
                await self._save_session()
            except Exception as e:  # noqa: BLE001
                self.trace.log(
                    "session_save_error", turn=self._turn_count, error=str(e)
                )
                warnings.warn(f"Failed to save session: {e}", stacklevel=2)

    async def _execute_tool(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        dry_run: bool,
    ) -> ToolResult:
        """Execute a tool with full trace logging. Re-raises on crash."""
        self.trace.log(
            "tool_call",
            turn=self._turn_count,
            iteration=self._iteration_count,
            id=call_id,
            name=name,
            arguments=arguments,
            dry_run=dry_run,
        )
        start = time.monotonic()
        try:
            result = await self.registry.execute(name, arguments, dry_run=dry_run)
        except Exception as e:  # noqa: BLE001
            self.trace.log(
                "tool_result",
                turn=self._turn_count,
                iteration=self._iteration_count,
                id=call_id,
                name=name,
                dry_run=dry_run,
                success=False,
                error=f"Tool error: {e}",
                exception_type=type(e).__name__,
                traceback=traceback.format_exc(),
                duration=time.monotonic() - start,
            )
            raise
        self.trace.log(
            "tool_result",
            turn=self._turn_count,
            iteration=self._iteration_count,
            id=call_id,
            name=name,
            dry_run=dry_run,
            success=result.success,
            output=result.output,
            error=result.error,
            requires_confirmation=result.requires_confirmation,
            duration=time.monotonic() - start,
        )
        return result

    async def _execute_remaining_tools(self) -> AsyncIterator[AgentEvent]:
        """Execute remaining placeholder tool calls from the last assistant turn.

        The assistant message already references every tool call in a multi-tool
        turn, but the calls after a confirmation pause only have placeholder
        results. Run those calls, replacing placeholders with real outputs, and
        stop if another call requires confirmation.
        """
        assistant = self._history.latest_assistant_with_tools()
        if assistant is None or assistant.tool_calls is None:
            return

        for start_idx, tc in enumerate(assistant.tool_calls):
            if tc["id"] not in self._history.pending_ids:
                continue

            name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
            try:
                result = await self._execute_tool(
                    tc["id"], name, arguments, dry_run=True
                )
            except Exception as e:  # noqa: BLE001
                error_message = f"Tool error: {e}"
                yield ErrorEvent(message=error_message)
                self._history.record_error(
                    assistant, start_idx, tc["id"], error_message
                )
                return
            if result.requires_confirmation:
                self._pending_tool = tc
                self._pending_since = time.monotonic()
            yield ToolResultEvent(
                id=tc["id"], name=name, result=result, arguments=arguments
            )
            if result.requires_confirmation:
                return
            self._history.record_result(
                tc["id"], result.output or result.error or ""
            )

    async def _conversation_loop(self) -> AsyncIterator[AgentEvent]:
        while self._iteration_count < self.config.llm.max_iterations:
            self._iteration_count += 1
            try:
                async for event in self._call_llm():
                    yield event
            except Exception as e:  # noqa: BLE001
                self.trace.log(
                    "llm_error",
                    turn=self._turn_count,
                    iteration=self._iteration_count,
                    error=str(e),
                    exception_type=type(e).__name__,
                    traceback=traceback.format_exc(),
                )
                yield ErrorEvent(message=f"LLM error: {e}")
                return

            last = self.messages[-1]
            if not last.tool_calls:
                break

            # If we've reached the iteration limit on an assistant message that
            # requests tool calls, cancel the calls instead of executing them.
            # OpenAI requires a matching ``role="tool"`` result for every
            # ``tool_call_id`` referenced by the assistant.
            if self._iteration_count >= self.config.llm.max_iterations:
                for tc in last.tool_calls:
                    self.messages.append(
                        Message(
                            role="tool",
                            content="Maximum iteration count reached; tool not executed.",
                            tool_call_id=tc["id"],
                        )
                    )
                self.trace.log(
                    "error",
                    turn=self._turn_count,
                    kind="max_iterations",
                    message="Maximum iteration count reached; remaining tool calls were cancelled.",
                    cancelled_tool_call_ids=[tc["id"] for tc in last.tool_calls],
                )
                yield ErrorEvent(
                    message="Maximum iteration count reached; remaining tool calls were cancelled."
                )
                break

            for idx, tc in enumerate(last.tool_calls):
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
                try:
                    result = await self._execute_tool(
                        tc["id"], name, arguments, dry_run=True
                    )
                except Exception as e:  # noqa: BLE001
                    error_message = f"Tool error: {e}"
                    yield ErrorEvent(message=error_message)
                    self._history.record_error(
                        last, idx, tc["id"], error_message
                    )
                    return
                if result.requires_confirmation:
                    self._pending_tool = tc
                    self._pending_since = time.monotonic()
                    # The assistant message already references every tool call in
                    # this turn, but only the calls before this one have results.
                    # Append placeholders for this pending call and the remaining
                    # calls so the message history stays valid for OpenAI. The
                    # pending call's placeholder is replaced by the real result
                    # when the user confirms it. Placeholders are installed
                    # before yielding the ToolResultEvent so confirmation handlers
                    # can update them immediately.
                    self._history.install_placeholders(
                        [t["id"] for t in last.tool_calls[idx:]]
                    )
                yield ToolResultEvent(
                    id=tc["id"], name=name, result=result, arguments=arguments
                )
                if result.requires_confirmation:
                    return
                self._history.record_result(
                    tc["id"], result.output or result.error or ""
                )

    async def apply_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Apply the tool that is currently pending confirmation.

        Validates that ``name`` and ``arguments`` match the stored pending tool
        call. Returns an error result and leaves the pending state untouched
        when they do not match or when execution fails, so the user can retry
        after transient failures.
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

        self.trace.log(
            "confirmation",
            turn=self._turn_count,
            decision="approved",
            tool_call_id=self._pending_tool["id"],
            name=pending_name,
            wait=self._pending_wait(),
        )
        try:
            result = await self._execute_tool(
                self._pending_tool["id"], name, arguments, dry_run=False
            )
        except Exception as e:  # noqa: BLE001
            error_message = f"Tool error: {e}"
            result = ToolResult(success=False, error=error_message)

        # Replace the placeholder tool result installed when the pending call
        # was first encountered, so the message history stays valid and shows
        # the actual outcome. Only clear the pending state on success so a
        # transient failure can be retried.
        self._history.record_result(
            self._pending_tool["id"], result.output or result.error or ""
        )
        if result.success:
            self._pending_tool = None
            self._pending_since = None
            self._confirmation_applied = True
        return result

    async def _call_llm(self) -> AsyncIterator[AgentEvent]:
        tool_definitions = self.registry.definitions()
        assistant_content = ""
        assistant_reasoning = ""
        assistant_signature: str | None = None
        tool_calls: list[dict[str, Any]] = []
        meta: CompletionMeta | None = None

        def _trace_request(body: dict[str, Any]) -> None:
            self.trace.log(
                "llm_request",
                turn=self._turn_count,
                iteration=self._iteration_count,
                body=body,
            )

        async for event in self.llm_client.chat(
            self.messages, tools=tool_definitions, on_request=_trace_request
        ):
            if isinstance(event, TextChunk):
                assistant_content += event.text
                yield TextDelta(text=event.text)
            elif isinstance(event, ThinkingChunk):
                assistant_reasoning += event.text
                if event.signature is not None:
                    assistant_signature = event.signature
                yield ThinkingDelta(text=event.text)
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
            elif isinstance(event, CompletionMeta):
                meta = event

        usage = meta.usage if meta else None
        self.trace.log(
            "llm_response",
            turn=self._turn_count,
            iteration=self._iteration_count,
            duration=meta.duration if meta else None,
            ttft=meta.ttft if meta else None,
            finish_reason=meta.finish_reason if meta else None,
            usage=usage,
            cached_tokens=_extract_cached_tokens(usage),
            content_chars=len(assistant_content),
            reasoning_chars=len(assistant_reasoning),
            tool_calls=[
                {"id": tc["id"], "name": tc["function"]["name"]}
                for tc in tool_calls
            ],
        )

        self.messages.append(
            Message(
                role="assistant",
                content=assistant_content or None,
                tool_calls=tool_calls if tool_calls else None,
                reasoning=assistant_reasoning or None,
                reasoning_signature=assistant_signature,
            )
        )

    def _save_session_sync(self) -> None:
        # Rewrite the whole session on every save. This guarantees that in-place
        # updates to placeholder messages (e.g. after a confirmation) are
        # reflected on disk, and it is still cheap for MVP-sized conversations.
        if not self._meta.title:
            self._meta.title = derive_title(self.messages)
        save_session(self._session_file, self._meta, self.messages)

    async def _save_session(self) -> None:
        await asyncio.to_thread(self._save_session_sync)
