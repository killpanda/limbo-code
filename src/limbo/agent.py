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
from limbo.compaction import (
    CompactionConfig,
    build_summary_prompt,
    estimate_tokens,
    find_split_point,
    make_summary_message,
    should_compact,
)
from limbo.config import Config
from limbo.history import ToolHistory
from limbo.history import repair as repair_history
from limbo.llm.catalog import resolve_model
from limbo.llm.client import LLMClient
from limbo.llm.retry import friendly_message
from limbo.models import (
    CompletionMeta,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
    ToolResult,
)
from limbo.sessions import (
    CompactionRecord,
    SessionMeta,
    derive_title,
    load_session,
    save_session,
)
from limbo.skills import discover_skills, format_skills_for_prompt
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


@dataclass(frozen=True)
class CompactionEvent:
    """Outcome of one compaction attempt (auto at loop top, or /compact)."""

    compacted: bool
    trigger: str  # "auto" | "manual"
    before_tokens: int = 0
    after_estimate: int | None = None
    summary_chars: int = 0
    # Why nothing happened (history too short, summary failed, ...).
    reason: str | None = None
    # Non-fatal follow-up note (e.g. still over budget after compacting).
    warning: str | None = None


@dataclass(frozen=True)
class UsageUpdate:
    """Cumulative token usage for the session, emitted after each LLM call."""

    total_tokens: int


AgentEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallRequest
    | ToolResultEvent
    | ErrorEvent
    | CompactionEvent
    | UsageUpdate
)


def _extract_prompt_tokens(usage: dict[str, Any] | None) -> int | None:
    """Normalize provider-specific prompt-size counters.

    OpenAI reports ``prompt_tokens``; Anthropic reports ``input_tokens``
    (which excludes cache reads/creation, so those are added back).
    """
    if not usage:
        return None
    prompt = usage.get("prompt_tokens")
    if isinstance(prompt, int):
        return prompt
    input_tokens = usage.get("input_tokens")
    if isinstance(input_tokens, int):
        total = input_tokens
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            extra = usage.get(key)
            if isinstance(extra, int):
                total += extra
        return total
    return None


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


def _extract_total_tokens(usage: dict[str, Any] | None) -> int | None:
    """Normalize provider-specific total-token counters.

    OpenAI-compatible providers report ``total_tokens``; Anthropic reports
    separate ``input_tokens`` / ``output_tokens``.
    """
    if not usage:
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if isinstance(input_tokens, int) or isinstance(output_tokens, int):
        return (input_tokens or 0) + (output_tokens or 0)
    return None


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
        self._iteration_count = 0
        self._last_finish_reason: str | None = None
        self._session_total_tokens = 0
        self._init_system_message()

        # Auto-compaction state (LIM-14). ``_last_prompt_tokens`` is the
        # real prompt size reported by the last response; ``_usage_watermark``
        # is len(messages) at that moment, so the next prompt can be
        # estimated as last + estimate(messages appended since).
        self._context_window = resolve_model(config.llm.model).context_window
        self._compaction_config = self._build_compaction_config(config)
        self._last_prompt_tokens: int | None = None
        self._usage_watermark: int = 0
        self._compactions: list[CompactionRecord] = []
        self._previous_summary: str | None = None
        self._auto_compaction_failed_this_turn = False
        self._auto_compact_noop_notified = False

        self._session_dir = session_dir or Path.home() / ".limbo" / "sessions"
        if resume is not None:
            self._meta, history, self._compactions = load_session(resume)
            self._session_file = resume
            self.messages.extend(repair_history(history))
            # Restore the iterative-summary chain across sessions.
            self._previous_summary = next(
                (c.summary for c in reversed(self._compactions) if c.summary),
                None,
            )
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

    def _build_compaction_config(self, config: Config) -> CompactionConfig:
        """Map [compaction] settings, with a cross-check against the window.

        reserve + keep_recent must leave headroom inside the model's context
        window; otherwise compaction could succeed yet stay over threshold
        and re-burn a summary call every iteration. Misconfiguration falls
        back to defaults with a warning; a window too small even for the
        defaults disables compaction outright.
        """
        reserve = config.compaction.reserve_tokens
        keep_recent = config.compaction.keep_recent_tokens
        enabled = config.compaction.enabled
        if enabled and reserve + keep_recent >= self._context_window:
            warnings.warn(
                f"compaction reserve_tokens({reserve}) + "
                f"keep_recent_tokens({keep_recent}) >= context window "
                f"({self._context_window}); reset to defaults "
                f"(16384/20000).",
                stacklevel=2,
            )
            reserve, keep_recent = 16_384, 20_000
            if reserve + keep_recent >= self._context_window:
                warnings.warn(
                    f"context window ({self._context_window}) too small for "
                    "auto-compaction; disabled.",
                    stacklevel=2,
                )
                enabled = False
        return CompactionConfig(
            enabled=enabled,
            reserve_tokens=reserve,
            keep_recent_tokens=keep_recent,
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
        # Load optional context files (project AGENTS.md, global
        # ~/.limbo/AGENTS.md) wrapped in XML boundary tags.
        context_files: list[tuple[str, str]] = []
        project_md = self.workdir / "AGENTS.md"
        if project_md.is_file():
            try:
                context_files.append((str(project_md), project_md.read_text(encoding="utf-8")))
            except OSError:
                pass
        global_md = Path.home() / ".limbo" / "AGENTS.md"
        if global_md.is_file():
            try:
                context_files.append((str(global_md), global_md.read_text(encoding="utf-8")))
            except OSError:
                pass
        if context_files:
            block = "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
            for file_path, content in context_files:
                block += (
                    f'<project_instructions path="{file_path}">\n\n'
                    f"{content}\n\n</project_instructions>\n\n"
                )
            block += "</project_context>\n"
            content_parts.append(block)
        # Skills catalog (progressive disclosure): name/description/location
        # only; the model loads full SKILL.md files with read on demand.
        # Only injected when the read tool is available.
        if self.registry.get("read") is not None:
            skills_block = format_skills_for_prompt(discover_skills(self.workdir))
            if skills_block:
                content_parts.append(skills_block)
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

    def _log_turn_end(self) -> None:
        if self._turn_start is None:
            return
        self.trace.log(
            "turn_end",
            turn=self._turn_count,
            duration=time.monotonic() - self._turn_start,
            iterations=self._iteration_count,
            status="completed",
        )
        self._turn_start = None

    async def run(self, user_input: str) -> AsyncIterator[AgentEvent]:
        # Reset per-user-turn state.
        self._iteration_count = 0
        self._auto_compaction_failed_this_turn = False
        self._auto_compact_noop_notified = False
        self._turn_count += 1
        self._turn_start = time.monotonic()

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

    async def _execute_tool(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with full trace logging. Re-raises on crash."""
        self.trace.log(
            "tool_call",
            turn=self._turn_count,
            iteration=self._iteration_count,
            id=call_id,
            name=name,
            arguments=arguments,
        )
        start = time.monotonic()
        try:
            result = await self.registry.execute(name, arguments)
        except Exception as e:  # noqa: BLE001
            self.trace.log(
                "tool_result",
                turn=self._turn_count,
                iteration=self._iteration_count,
                id=call_id,
                name=name,
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
            success=result.success,
            output=result.output,
            error=result.error,
            duration=time.monotonic() - start,
        )
        return result

    async def _conversation_loop(self) -> AsyncIterator[AgentEvent]:
        while self._iteration_count < self.config.llm.max_iterations:
            self._iteration_count += 1
            # Loop-top check: history is consistent here (all tool results
            # from the previous batch are recorded) and no request has been
            # sent yet, so this is the safest point to compact.
            if (
                not self._auto_compaction_failed_this_turn
                and should_compact(
                    self._estimated_next_prompt_tokens(),
                    self._context_window,
                    self._compaction_config,
                )
            ):
                # Defensive: a bug in compaction must never kill the turn.
                try:
                    async for event in self.compact(trigger="auto"):
                        yield event
                except Exception as e:  # noqa: BLE001
                    self._auto_compaction_failed_this_turn = True
                    self.trace.log(
                        "compaction_error",
                        turn=self._turn_count,
                        iteration=self._iteration_count,
                        trigger="auto",
                        error=str(e),
                        exception_type=type(e).__name__,
                        traceback=traceback.format_exc(),
                    )
                    yield CompactionEvent(
                        compacted=False,
                        trigger="auto",
                        reason=f"自动压缩异常：{e}",
                    )
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
                yield ErrorEvent(message=friendly_message(e) or f"LLM error: {e}")
                return

            last = self.messages[-1]
            if not last.tool_calls:
                break

            # Length-stop guard: the response was cut off by the output
            # token limit, so tool-call arguments may be truncated. Fail
            # the whole batch without executing it. OpenAI reports this as
            # "length"; Anthropic reports it as "max_tokens" (its
            # stop_reason is passed through verbatim by the client).
            if self._last_finish_reason in ("length", "max_tokens"):
                async for event in self._fail_truncated_tool_calls(last):
                    yield event
                continue

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
            async for event in self._execute_tool_calls(last):
                yield event

    async def _execute_tool_calls(self, assistant: Message) -> AsyncIterator[AgentEvent]:
        """Execute one batch of tool calls.

        Semantics (aligned with pi's agent loop):

        - Tool calls run concurrently when ``tools.parallel`` is enabled
          (default). ``ToolResultEvent``s stream out in *completion* order
          so a fast tool is never blocked behind a slow one; results are
          recorded into history in *source* order once the batch finishes,
          keeping session replay and traces readable.
        - A tool crash becomes an error ``ToolResult`` for that call only;
          sibling calls finish normally and the loop continues.
        """
        tool_calls = assistant.tool_calls or []
        results: dict[str, ToolResult] = {}

        if self.config.tools.parallel and len(tool_calls) > 1:
            async def run_one(tc: dict[str, Any]) -> tuple[dict[str, Any], ToolResult]:
                result = await self._execute_tool_safe(
                    tc["id"], tc["function"]["name"], tc["function"]["arguments"]
                )
                return tc, result

            futures = [asyncio.ensure_future(run_one(tc)) for tc in tool_calls]
            for future in asyncio.as_completed(futures):
                tc, result = await future
                results[tc["id"]] = result
                yield ToolResultEvent(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    result=result,
                    arguments=tc["function"]["arguments"],
                )
        else:
            for tc in tool_calls:
                result = await self._execute_tool_safe(
                    tc["id"], tc["function"]["name"], tc["function"]["arguments"]
                )
                results[tc["id"]] = result
                yield ToolResultEvent(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    result=result,
                    arguments=tc["function"]["arguments"],
                )

        # Record results in assistant source order after the batch completes.
        for tc in tool_calls:
            result = results[tc["id"]]
            self._history.record_result(
                tc["id"], result.output or result.error or ""
            )

    async def _execute_tool_safe(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute one tool, converting crashes into error results."""
        try:
            return await self._execute_tool(call_id, name, arguments)
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"Tool error: {e}")

    async def _fail_truncated_tool_calls(
        self, assistant: Message
    ) -> AsyncIterator[AgentEvent]:
        """Fail a whole batch without executing it (length-stop guard).

        A ``finish_reason == "length"`` response was cut off by the output
        token limit, so any tool call in it may carry truncated arguments
        that still parse. None of them are safe to execute; return an error
        result for each so the model can re-issue them (pi does the same).
        """
        message = (
            "Tool call was not executed: the response hit the output token "
            "limit, so its arguments may be truncated. Re-issue the tool "
            "call with complete arguments."
        )
        self.trace.log(
            "error",
            turn=self._turn_count,
            kind="length_stop",
            message="Response hit the output token limit; tool calls were not executed.",
            cancelled_tool_call_ids=[tc["id"] for tc in assistant.tool_calls or []],
        )
        for tc in assistant.tool_calls or []:
            name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
            result = ToolResult(success=False, error=message)
            yield ToolResultEvent(
                id=tc["id"], name=name, result=result, arguments=arguments
            )
            self._history.record_result(tc["id"], message)

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
        self._last_finish_reason = meta.finish_reason if meta else None
        total = _extract_total_tokens(usage)
        if total:
            self._session_total_tokens += total
            yield UsageUpdate(total_tokens=self._session_total_tokens)
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
        # Record the real prompt size for the compaction trigger, with the
        # watermark past the assistant message just appended.
        prompt_tokens = _extract_prompt_tokens(usage)
        if prompt_tokens is not None:
            self._last_prompt_tokens = prompt_tokens
            self._usage_watermark = len(self.messages)

    # -- auto-compaction (LIM-14) --------------------------------------------

    def _estimated_next_prompt_tokens(self) -> int:
        """Estimate the size of the *next* request's prompt.

        With real usage: last prompt + estimated tokens of everything
        appended since (new user input, assistant messages, tool results).
        This catches the classic overflow case — a huge paste between turns
        — that a stale last-response figure alone would miss. Without usage
        (first iteration, or a provider that reports none): full estimate.
        """
        if self._last_prompt_tokens is None:
            return estimate_tokens(self.messages)
        return self._last_prompt_tokens + estimate_tokens(
            self.messages[self._usage_watermark :]
        )

    async def _generate_summary(self, prompt: list[Message]) -> str:
        """One-shot summarization call on the current client (no tools)."""

        def _trace_request(body: dict[str, Any]) -> None:
            self.trace.log(
                "llm_request",
                turn=self._turn_count,
                iteration=self._iteration_count,
                kind="compaction",
                body=body,
            )

        text = ""
        meta: CompletionMeta | None = None
        async for event in self.llm_client.chat(
            prompt, tools=[], on_request=_trace_request
        ):
            if isinstance(event, TextChunk):
                text += event.text
            elif isinstance(event, CompletionMeta):
                meta = event
        self.trace.log(
            "llm_response",
            turn=self._turn_count,
            iteration=self._iteration_count,
            kind="compaction",
            duration=meta.duration if meta else None,
            finish_reason=meta.finish_reason if meta else None,
            usage=meta.usage if meta else None,
            content_chars=len(text),
        )
        return text

    async def compact(self, trigger: str = "auto") -> AsyncIterator[AgentEvent]:
        """Compact history: summarize the old region, keep the recent tail.

        Shared by the loop-top auto trigger (``trigger="auto"``) and the
        ``/compact`` slash command (``trigger="manual"``). On any failure
        the history is left untouched; an auto failure suppresses further
        auto attempts for the rest of the turn (the guard resets in
        :meth:`run`), while ``/compact`` can always be retried by hand.
        """
        cfg = self._compaction_config
        before = self._estimated_next_prompt_tokens()
        split = find_split_point(self.messages, cfg.keep_recent_tokens)
        if split is None:
            if trigger == "auto":
                # Retried every iteration on purpose (a later assistant
                # message may restore a safe cut) but reported at most once
                # per turn to avoid flooding the chat.
                if self._auto_compact_noop_notified:
                    return
                self._auto_compact_noop_notified = True
                reason = "最近消息过大或历史太短，无法安全切分保留边界，本次跳过自动压缩"
            else:
                reason = "历史太短，无需压缩"
            yield CompactionEvent(
                compacted=False,
                trigger=trigger,
                before_tokens=before,
                reason=reason,
            )
            return
        # Freeze the title before the first real user message can be
        # summarized away (titles are re-derived on every save).
        if not self._meta.title:
            self._meta.title = derive_title(self.messages)
        to_summarize = self.messages[1:split]
        prompt = build_summary_prompt(to_summarize, self._previous_summary)
        start = time.monotonic()
        summary = ""
        error: str | None = None
        try:
            summary = await self._generate_summary(prompt)
            if not summary.strip():
                error = "empty response"
        except Exception as e:  # noqa: BLE001
            error = str(e)
        if error is not None:
            if trigger == "auto":
                self._auto_compaction_failed_this_turn = True
            self.trace.log(
                "compaction_error",
                turn=self._turn_count,
                iteration=self._iteration_count,
                trigger=trigger,
                error=error,
            )
            yield CompactionEvent(
                compacted=False,
                trigger=trigger,
                before_tokens=before,
                reason=f"摘要生成失败：{error}",
            )
            return

        system_msg = self.messages[0]
        kept = self.messages[split:]
        self.messages = [system_msg, make_summary_message(summary), *kept]
        after = estimate_tokens(self.messages)
        record = CompactionRecord(
            id=secrets.token_hex(8),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            first_kept_entry_id=kept[0].id,
            summary=summary.strip(),
            tokens_before=before,
            tokens_after_estimate=after,
            trigger=trigger,
        )
        self._compactions.append(record)
        self._previous_summary = summary.strip()
        # Usage figures predate the rewrite; fall back to full estimation
        # until the next response reports real prompt tokens.
        self._last_prompt_tokens = None
        self._usage_watermark = len(self.messages)
        warning = None
        if should_compact(after, self._context_window, cfg):
            # No re-loop: one warning, the turn continues.
            warning = "压缩后上下文仍接近上限，建议尽快 /new 开新会话"
        self.trace.log(
            "compaction",
            turn=self._turn_count,
            iteration=self._iteration_count,
            trigger=trigger,
            tokens_before=before,
            tokens_after_estimate=after,
            split_index=split,
            summary_chars=len(summary),
            duration=time.monotonic() - start,
        )
        yield CompactionEvent(
            compacted=True,
            trigger=trigger,
            before_tokens=before,
            after_estimate=after,
            summary_chars=len(summary),
            warning=warning,
        )

    def _save_session_sync(self) -> None:
        # Rewrite the whole session on every save. It is cheap for MVP-sized
        # conversations.
        if not self._meta.title:
            self._meta.title = derive_title(self.messages)
        save_session(self._session_file, self._meta, self.messages, self._compactions)

    async def _save_session(self) -> None:
        await asyncio.to_thread(self._save_session_sync)
