"""Agent conversation loop."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import secrets
import threading
import time
import traceback
import warnings
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from limbo import __version__
from limbo.attachments import build_user_content
from limbo.compaction import (
    CompactionConfig,
    build_summary_prompt,
    find_split_point,
    is_summary_message,
    make_summary_message,
    should_compact,
)
from limbo.config import Config
from limbo.history import ToolHistory
from limbo.history import recover as recover_history
from limbo.llm.catalog import resolve_base_url, resolve_model
from limbo.llm.client import LLMClient, RequestHook
from limbo.llm.retry import friendly_message, is_context_overflow
from limbo.llm.usage import (
    PromptSizeEstimator,
    UsageTotals,
    estimate_tokens,
    normalize_usage,
)
from limbo.models import (
    Attachment,
    CompletionMeta,
    Message,
    TextChunk,
    ThinkingChunk,
    ToolCallEvent,
    ToolResult,
)
from limbo.prompt import build_system_prompt
from limbo.sessions import (
    CompactionRecord,
    QueuedMessageRecord,
    SessionMeta,
    UsageEntry,
    append_session_messages,
    append_session_records,
    compaction_record,
    dequeued_record,
    derive_title,
    load_session_full,
    meta_patch_record,
    queued_record,
    rewrite_meta_line,
    save_session,
    tool_intent_record,
)
from limbo.steer import FOLLOW_UP, STEER, SteerItem, SteerMode, SteerQueue
from limbo.tools.registry import ToolRegistry
from limbo.trace import TRACE_SCHEMA, TraceLogger, trace_path_for, utc_now_iso

# Consecutive length-stop rejections tolerated before the turn is aborted.
# A truncated response whose tool calls were refused is almost always the
# model re-issuing the same oversized arguments (LIM-32): without a guard
# it burns the whole iteration budget re-sending what can never fit.
_MAX_CONSECUTIVE_LENGTH_STOPS = 3

# Placeholder result for tool calls that never started because the user
# interrupted the batch (RFC LIM-53). Only ever used for calls that
# genuinely never ran, so history and disk cannot diverge.
_NOT_EXECUTED_MESSAGE = "未执行：用户打断（ESC）"


class _StreamInterrupted(Exception):
    """Internal: the LLM stream pump was cancelled by interrupt()."""


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
    trigger: str  # "auto" | "manual" | "overflow"
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


@dataclass(frozen=True)
class SteerEvent:
    """A queued steer message was injected into the conversation history.

    Emitted by every consumption point (loop-top steer, turn-end follow-up,
    run()-head fallback) so the UI can flip the matching queued card by id.
    """

    id: str
    text: str


@dataclass(frozen=True)
class InterruptEvent:
    """The user interrupted the current activity with ESC (RFC LIM-53)."""

    phase: str  # "llm_stream" | "tool_batch" | "compact"


AgentEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallRequest
    | ToolResultEvent
    | ErrorEvent
    | CompactionEvent
    | UsageUpdate
    | SteerEvent
    | InterruptEvent
)


def _resolved_llm_trace_fields(config: Config) -> dict[str, Any]:
    """Resolved provider identity for trace events.

    The raw ``[llm]`` config values alone can't tell which endpoint/dialect
    actually served a session (provider overrides and catalog defaults win
    over them), so trace events record the resolved spec alongside.
    """
    spec = resolve_model(config.llm.model)
    return {
        "model": spec.id,
        "provider": spec.provider.id,
        "api": spec.provider.api,
        "base_url": resolve_base_url(spec, config),
        "context_window": spec.context_window,
        "max_tokens": config.llm.max_tokens or spec.max_tokens,
        "thinking_effort": config.llm.thinking_effort,
    }


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
        # Steer queue (RFC LIM-20): user messages submitted mid-turn. The
        # queue owns queueing semantics; this loop owns drain timing.
        self._steer_queue = SteerQueue()
        # Interrupt state (RFC LIM-53): ESC cancels the in-flight LLM
        # stream / tool batch / compact summary at the nearest
        # history-consistency point.
        self._interrupt_requested = False
        self._turn_interrupted = False
        self._stream_task: asyncio.Task[Any] | None = None
        self._stream_cut = False
        self._iteration_count = 0
        self._last_finish_reason: str | None = None
        self._consecutive_length_stops = 0
        self._init_system_message()

        # Auto-compaction state (LIM-14). The estimator holds the real
        # prompt size reported by the last response plus a watermark over
        # the message list, so the next prompt can be estimated as
        # last + estimate(messages appended since).
        self._context_window = resolve_model(config.llm.model).context_window
        # Vision gate for image attachments (RFC LIM-17): only models
        # flagged vision=True in the catalog receive image blocks; others
        # get a path-reference degradation instead.
        self._vision = resolve_model(config.llm.model).vision
        self._compaction_config = self._build_compaction_config(config)
        self._prompt_estimator = PromptSizeEstimator()
        self._compactions: list[CompactionRecord] = []
        self._previous_summary: str | None = None
        # Incremental session persistence: how many messages are already on
        # disk (append-only fast path), and whether the next save must do a
        # full rewrite instead (history reshaped by compaction, or a resume
        # whose repair() inserted placeholder tool messages).
        self._session_saved_count = 0
        self._session_saved_meta: str | None = None
        # The title on the file's meta line (listing reads only that line).
        self._session_saved_title: str | None = None
        self._session_full_rewrite_needed = False
        # Compaction records not yet appended (non-destructive compaction:
        # the record goes into the journal, the summarized messages stay).
        self._pending_compactions: list[CompactionRecord] = []
        # Messages summarized away before they reached disk: written ahead
        # of the compaction record so the journal still holds them.
        self._pending_summarized: list[Message] = []
        # Queue items injected or cancelled since the last save; their
        # "dequeued" record is written after the messages that carry them.
        self._pending_dequeued: list[str] = []
        # Serializes session-file writes: checkpoints run in a worker thread
        # while meta patches (goal changes) are written from the loop.
        self._session_io_lock = threading.Lock()
        self._checkpoint_failed_this_turn = False
        self._overflow_recovered_this_turn = False
        self._last_run_failed = False
        self._auto_compaction_failed_this_turn = False
        self._auto_compact_noop_notified = False

        self._session_dir = session_dir or Path.home() / ".limbo" / "sessions"
        # Read-only tool calls interrupted by a crash, re-run at the start
        # of the next turn (their placeholder result is then replaced).
        self._pending_reruns: list[dict[str, Any]] = []
        if resume is not None:
            loaded = load_session_full(resume)
            self._meta, history, self._compactions = (
                loaded.meta,
                loaded.messages,
                loaded.compactions,
            )
            self._session_file = resume
            recovery = recover_history(
                history,
                started=(
                    {i.call_id for i in loaded.intents}
                    if self._meta.tool_intents
                    else None
                ),
                is_safe=lambda name: self.registry.replay_policy(name) == "safe",
            )
            self.messages.extend(recovery.messages)
            self._pending_reruns = recovery.rerun
            if loaded.queue:
                # Messages still queued when the process died (pi-durable's
                # persisted inbox): back into the queue, already journaled.
                self._steer_queue.restore(
                    [
                        SteerItem(
                            id=q.id,
                            text=q.text,
                            attachments=q.attachments,
                            mode=FOLLOW_UP if q.mode == FOLLOW_UP else STEER,
                        )
                        for q in loaded.queue
                    ]
                )
            self._session_saved_meta = self._meta_snapshot()
            # From here on every call is journaled, so a later crash can be
            # recovered precisely even if this session predates intents
            # (the flag reaches the file as a meta patch on the next save).
            self._meta.tool_intents = True
            self._session_saved_count = len(self.messages)
            self._session_saved_title = self._meta.title
            # Only a repaired history (placeholders inserted) must be
            # rewritten; otherwise the journal stays as it is and the next
            # save appends. The in-memory system message is regenerated and
            # the summary is rebuilt on load, so neither is ever appended.
            self._session_full_rewrite_needed = len(recovery.messages) != len(
                history
            ) - (1 if history and history[0].role == "system" else 0)
            # Restore the presentation mode: recover_history dropped the
            # loaded system message, so rebuild the fresh one in code mode
            # when the session was saved with it on.
            if self._meta.code_mode:
                self.registry.code_mode = True
                self._init_system_message()
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
                tool_intents=True,
            )

        # Trace log: full-fidelity JSONL record of the run, kept separate from
        # the resumable session file. Created lazily-tolerant: tracing must
        # never break the agent.
        self.trace = TraceLogger(trace_path_for(self._session_file))
        self._turn_count = 0
        self._turn_start: float | None = None
        self._turn_started_at: str | None = None
        self.trace.log(
            "session_start",
            trace_schema=TRACE_SCHEMA,
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
            resolved=_resolved_llm_trace_fields(config),
        )
        self._apply_prompt_cache_key()
        self.registry.trace_log = self._trace_sub_call

    def _trace_sub_call(self, event: str, **fields: Any) -> None:
        self.trace.log(
            event,
            turn=self._turn_count,
            iteration=self._iteration_count,
            kind="sub_call",
            **fields,
        )

    def _apply_prompt_cache_key(self) -> None:
        """Route this session's requests to one provider cache (Responses)."""
        if hasattr(self.llm_client, "prompt_cache_key"):
            self.llm_client.prompt_cache_key = self._meta.id

    # -- usage ledger ----------------------------------------------------------

    @property
    def session_total_tokens(self) -> int:
        """Cumulative tokens across every model this session has used."""
        return sum(entry.total_tokens for entry in self._meta.usage.values())

    @property
    def usage_ledger(self) -> dict[str, UsageEntry]:
        return dict(self._meta.usage)

    def record_usage(self, usage: dict[str, Any] | None) -> UsageTotals:
        """Add one completion's usage to the ledger (``provider/model``).

        Persisted with the next save as a ``meta_patch``. Public so side
        calls made outside the loop (``/btw``) are counted too.
        """
        totals = normalize_usage(usage)
        if not usage:
            return totals
        model = self.config.llm.model
        key = f"{resolve_model(model).provider.id}/{model}"
        entry = self._meta.usage.setdefault(key, UsageEntry())
        entry.requests += 1
        entry.prompt_tokens += totals.prompt_tokens or 0
        entry.cached_tokens += totals.cached_tokens or 0
        entry.total_tokens += totals.total_tokens or 0
        return totals

    def close(self) -> None:
        """Release owned resources (the trace log file handle).

        Call when this agent is replaced (``/new``, resume) or torn down so
        the append-only trace file descriptor is not leaked across sessions.
        The LLM client is owned by the UI layer and closed separately.
        Idempotent: safe to call more than once.
        """
        self.trace.close()

    def update_llm(self, llm_client: LLMClient) -> None:
        """Swap the LLM client after a /model switch.

        ``config.llm.model`` must already hold the new model id (the spec is
        re-resolved here) so the context window, vision gate, and compaction
        budget all follow the new model.
        """
        spec = resolve_model(self.config.llm.model)
        self.llm_client = llm_client
        self._context_window = spec.context_window
        self._vision = spec.vision
        self._compaction_config = self._build_compaction_config(self.config)
        self._apply_prompt_cache_key()
        self.trace.log(
            "model_switch",
            turn=self._turn_count,
            resolved=_resolved_llm_trace_fields(self.config),
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
    def session_path(self) -> Path:
        """Path of the current session JSONL file."""
        return self._session_file

    @property
    def session_meta(self) -> SessionMeta:
        """Metadata of the current session (title filled in on first save)."""
        if not self._meta.title:
            self._meta.title = derive_title(self.messages)
        return self._meta

    @property
    def context_window(self) -> int:
        """Context window of the active model, in tokens."""
        return self._context_window

    def context_tokens(self) -> int:
        """Estimated prompt size of the next request (the UI context meter)."""
        return self._estimated_next_prompt_tokens()

    def _init_system_message(self) -> None:
        prompt = build_system_prompt(self.registry, self.workdir)
        if self.messages and self.messages[0].role == "system":
            # Code Mode toggles rebuild the system prompt in place (the
            # SDK section changes with the presentation).
            self.messages[0] = Message(role="system", content=prompt)
        else:
            self.messages.append(Message(role="system", content=prompt))

    @property
    def messages(self) -> list[Message]:
        """The conversation history (owned by the tool-call bookkeeper)."""
        return self._history.messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._history.messages = value

    # -- code mode (deepseek-harness parity) ---------------------------------

    @property
    def code_mode(self) -> bool:
        """Whether the model-facing catalog is collapsed to ``run_code``."""
        return self.registry.code_mode

    def set_code_mode(self, enabled: bool) -> bool:
        """Toggle Code Mode: rebuild the system prompt (SDK section).

        Returns False when the mode did not change (the caller can report
        "already on/off" instead of a fake switch). The UI owns the busy
        guard — a mid-turn switch would corrupt the in-flight request.
        """
        if self.registry.code_mode == enabled:
            return False
        self.registry.code_mode = enabled
        self._init_system_message()
        self.trace.log(
            "code_mode", turn=self._turn_count, enabled=enabled
        )
        return True

    # -- steer queue (LIM-20) -------------------------------------------------

    def steer(
        self,
        text: str,
        attachments: list[Attachment] | None = None,
        mode: SteerMode = STEER,
    ) -> str:
        """Queue a user message for the running task.

        ``steer`` (default) is injected at the next consistency point of the
        running turn; ``follow_up`` waits until the task is done. The item
        is journaled immediately so a crash keeps it. Non-blocking; safe to
        call from the UI at any time. Returns the queued item's id (the UI
        binds its queued card to it).
        """
        with self._session_io_lock:
            item = self._steer_queue.offer(text, attachments, mode)
            if self._session_file.exists() and not self._session_full_rewrite_needed:
                try:
                    append_session_records(
                        self._session_file, [queued_record(_queue_record(item))]
                    )
                except OSError as e:
                    # The next full rewrite journals the whole queue.
                    self._session_full_rewrite_needed = True
                    self.trace.log(
                        "session_save_error",
                        turn=self._turn_count,
                        phase="queue",
                        error=str(e),
                    )
            # Otherwise the first (full) save journals the queue.
        return item.id

    def _dequeue(self, items: list[SteerItem]) -> list[SteerItem]:
        """Note items that left the queue; journaled with the next save."""
        self._pending_dequeued.extend(item.id for item in items)
        return items

    def cancel_steer(self, item_id: str) -> bool:
        """Cancel a still-queued steer message.

        Returns False when the item was already drained (injected or being
        injected — past the boundary it is part of history and cannot be
        retracted) or the id is unknown.
        """
        with self._session_io_lock:
            item = self._steer_queue.cancel(item_id)
            if item is None:
                return False
            self._dequeue([item])
        self.trace.log(
            "steer_cancelled",
            turn=self._turn_count,
            id=item.id,
            text=item.text,
        )
        return True

    def cancel_latest_steer(self) -> str | None:
        """Cancel the newest queued steer message (Esc path)."""
        with self._session_io_lock:
            item = self._steer_queue.cancel_latest()
            if item is None:
                return None
            self._dequeue([item])
        self.trace.log(
            "steer_cancelled", turn=self._turn_count, id=item.id, text=item.text
        )
        return item.id

    def drain_steer(
        self, modes: tuple[SteerMode, ...] = (STEER, FOLLOW_UP)
    ) -> list[SteerItem]:
        """Remove and return the queued items of ``modes`` in FIFO order.

        The agent's own drains (loop top, turn end, run head) take steers
        only; follow-ups wait for the pump's end-of-task drain.
        """
        with self._session_io_lock:
            return self._dequeue(self._steer_queue.drain(modes))

    @property
    def queued_items(self) -> list[SteerItem]:
        """Read-only snapshot of the queue (e.g. to re-render after resume)."""
        return list(self._steer_queue)

    # -- interrupt (LIM-53) ---------------------------------------------------

    def interrupt(self) -> None:
        """Interrupt the current turn / compact at the next consistency point.

        Non-blocking; safe to call from the UI at any time (a no-op when
        idle — the flag is reset at the next run()/compact() entry). Two
        layers, both required: the flag lets chunk-driven code stop
        gracefully with partial content, and cancelling the stream pump
        task covers stalled reads where no chunk ever arrives. In-flight
        bash process trees are killed immediately; file tools are not
        cancellable by design (they finish and record real results).
        """
        self._interrupt_requested = True
        task = self._stream_task
        if task is not None and not task.done():
            task.cancel()
        self.registry.cancel_active()

    @property
    def last_run_failed(self) -> bool:
        """Whether the last turn ended on an LLM error (after its retries).

        Reset at run() entry; the pump applies ``[queue] after_failed_run``
        to the queued messages when it is set.
        """
        return self._last_run_failed

    @property
    def was_interrupted(self) -> bool:
        """Whether the last turn ended by user interrupt.

        Reset at run() entry; consumed by the goal driver at the turn
        boundary (checked BEFORE the leftover-steer drain, RFC LIM-53).
        """
        return self._turn_interrupted

    def _mark_interrupted(self, phase: str, **extra: Any) -> InterruptEvent | None:
        """Consume a pending interrupt request; returns the event, once per turn."""
        if not self._interrupt_requested or self._turn_interrupted:
            return None
        self._turn_interrupted = True
        self.trace.log(
            "turn_interrupted",
            turn=self._turn_count,
            iteration=self._iteration_count,
            phase=phase,
            **extra,
        )
        return InterruptEvent(phase=phase)

    async def _anext_cancellable(self, stream: AsyncIterator[Any]) -> Any:
        """Next event of an LLM stream, interruptibly (RFC LIM-53).

        Raises StopAsyncIteration at stream end (as usual) and
        _StreamInterrupted when interrupt() cancelled the pump — the only
        way out of a stalled read where no chunk ever arrives.
        """
        task = asyncio.ensure_future(stream.__anext__())
        self._stream_task = task
        try:
            return await task
        except asyncio.CancelledError:
            if self._interrupt_requested:
                # interrupt() cancelled the pump (stalled-read layer).
                raise _StreamInterrupted from None
            # External cancellation of the turn worker itself (app quit):
            # must keep propagating — swallowing it would hang shutdown.
            raise
        finally:
            self._stream_task = None

    def has_pending_steer(self) -> bool:
        return len(self._steer_queue) > 0

    @property
    def queued_count(self) -> int:
        return len(self._steer_queue)

    def _inject_steer(self, item: SteerItem, *, followup: bool) -> SteerEvent:
        """Append a queued message to the history as a user message.

        Only ever called at consistency points (all tool results recorded,
        no request in flight): loop-top, turn-end, or run()-head.
        """
        content, images = self._build_user_content(item.text, item.attachments)
        self.messages.append(
            Message(role="user", content=content, attachments=images or None)
        )
        self.trace.log(
            "user_message",
            turn=self._turn_count,
            steer=not followup,
            followup=followup,
            id=item.id,
            content=content,
        )
        return SteerEvent(id=item.id, text=item.text)

    def _build_user_content(
        self, user_input: str, attachments: list[Attachment]
    ) -> tuple[str, list[Attachment]]:
        """Combine text with attachments (policy lives in limbo.attachments)."""
        return build_user_content(user_input, attachments, vision=self._vision)

    def _log_turn_end(self, status: str = "completed") -> None:
        if self._turn_start is None:
            return
        self.trace.log(
            "turn_end",
            turn=self._turn_count,
            started_at=self._turn_started_at,
            duration=time.monotonic() - self._turn_start,
            iterations=self._iteration_count,
            status=status,
        )
        self._turn_start = None

    async def run(
        self, user_input: str, attachments: list[Attachment] | None = None
    ) -> AsyncIterator[AgentEvent]:
        # Reset per-user-turn state.
        self._iteration_count = 0
        self._consecutive_length_stops = 0
        self._auto_compaction_failed_this_turn = False
        self._auto_compact_noop_notified = False
        self._checkpoint_failed_this_turn = False
        self._overflow_recovered_this_turn = False
        self._last_run_failed = False
        # RFC LIM-53: an interrupt requested while idle must not leak into
        # this turn (one of the two reset points; compact() is the other).
        self._interrupt_requested = False
        self._turn_interrupted = False
        self._turn_count += 1
        self._turn_start = time.monotonic()
        self._turn_started_at = utc_now_iso()

        # Fallback drain (RFC LIM-20): leftovers from a previous turn (error
        # exit, max-iterations cancellation, or the exit-window race) are
        # injected BEFORE this turn's input so submission order is preserved.
        for item in self.drain_steer((STEER,)):
            yield self._inject_steer(item, followup=True)
        async for event in self._rerun_interrupted():
            yield event

        content, images = self._build_user_content(user_input, attachments or [])
        self.messages.append(
            Message(role="user", content=content, attachments=images or None)
        )
        self.trace.log(
            "user_message", turn=self._turn_count, content=content
        )

        completed = False
        try:
            async for event in self._conversation_loop():
                yield event
            completed = True
        finally:
            # A turn killed mid-stream (worker cancelled on app quit, or the
            # generator closed) never logs llm_response/llm_error — the
            # turn_end status is the only trace of what happened, so it
            # must not claim "completed". A turn the user interrupted
            # (RFC LIM-53) breaks gracefully, so without the flag check it
            # would be logged "completed" — trace must distinguish both.
            status = (
                "interrupted"
                if (self._turn_interrupted or not completed)
                else "completed"
            )
            self._log_turn_end(status)
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
        await self._commit_intent(call_id, name)
        self.trace.log(
            "tool_call",
            turn=self._turn_count,
            iteration=self._iteration_count,
            id=call_id,
            name=name,
            arguments=arguments,
        )
        started_at = utc_now_iso()
        start = time.monotonic()
        try:
            result = await self.registry.execute(name, arguments, call_id=call_id)
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
                started_at=started_at,
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
            started_at=started_at,
            duration=time.monotonic() - start,
        )
        return result

    async def _commit_intent(self, call_id: str, name: str) -> None:
        """Journal a tool call before it runs (pi-durable's effect sandwich).

        On resume, a dangling call with a committed intent may have
        partially run, one without never started. Best effort: a failed
        write is reported like a failed checkpoint and the tool still runs.
        """
        record = tool_intent_record(
            call_id, name, self.registry.replay_policy(name)
        )

        def _sync() -> None:
            with self._session_io_lock:
                append_session_records(self._session_file, [record])

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:  # noqa: BLE001
            self._report_save_error("intent", e)

    async def _rerun_interrupted(self) -> AsyncIterator[AgentEvent]:
        """Re-run read-only calls a crash interrupted (resume only).

        Their placeholder results were inserted by recover_history; the
        real result replaces the placeholder in place, so history order is
        unchanged. Unsafe calls are never re-run — the model was told they
        may have partially run and decides itself.
        """
        reruns, self._pending_reruns = self._pending_reruns, []
        if not reruns:
            return
        self.trace.log(
            "tool_recovery",
            turn=self._turn_count,
            rerun_tool_call_ids=[tc["id"] for tc in reruns],
        )
        for tc in reruns:
            name = tc["function"]["name"]
            arguments = tc["function"]["arguments"]
            yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
            result = await self._execute_tool_safe(tc["id"], name, arguments)
            yield ToolResultEvent(
                id=tc["id"], name=name, result=result, arguments=arguments
            )
            content, attachments = self._gate_tool_attachments(result)
            self._history.record_result(tc["id"], content, attachments)
        # Results replaced messages that may already be on disk.
        self._session_full_rewrite_needed = True

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
            # RFC LIM-53: an interrupt consumed inside auto-compaction ends
            # the turn here — before the steer drain, so queued messages
            # stay queued.
            if self._turn_interrupted:
                break
            # Loop-top steer drain (RFC LIM-20): history is consistent here
            # (previous batch's tool results are all recorded) and no request
            # is in flight — the same safety argument as loop-top compaction.
            for item in self.drain_steer((STEER,)):
                yield self._inject_steer(item, followup=False)
            # Durable progress: the user input, injected steers, and any
            # compaction above are on disk before the request goes out.
            await self._checkpoint()
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
                    context_overflow=is_context_overflow(e),
                )
                if is_context_overflow(e) and not self._overflow_recovered_this_turn:
                    # pi-durable's overflow recovery: compact, then resend
                    # once. A second overflow in the same turn surfaces.
                    self._overflow_recovered_this_turn = True
                    compacted = False
                    async for event in self.compact(trigger="overflow"):
                        if isinstance(event, CompactionEvent):
                            compacted = event.compacted
                        yield event
                    if compacted and not self._turn_interrupted:
                        # The rejected request does not use up an iteration.
                        self._iteration_count -= 1
                        continue
                    if self._turn_interrupted:
                        break
                self._last_run_failed = True
                yield ErrorEvent(message=friendly_message(e) or f"LLM error: {e}")
                return
            # The response (and its tool calls) is on disk before any tool
            # runs, so a crash mid-batch leaves the calls for repair.
            await self._checkpoint()

            last = self.messages[-1]
            if not last.tool_calls:
                # A text response cut off by the output limit must not look
                # like a complete answer (pi shows the same warning).
                if self._last_finish_reason in ("length", "max_tokens"):
                    yield ErrorEvent(
                        message=(
                            "Model stopped because it reached the output token "
                            f"limit (max_tokens={self._effective_max_tokens()}); "
                            "the response above may be incomplete."
                        )
                    )
                # Turn-end follow-up drain (RFC LIM-20): messages that arrived
                # as the final response streamed keep the turn alive as a
                # follow-up — a fresh user input, so the iteration budget
                # resets like run() does per input.
                # RFC LIM-53: an interrupted turn ends here instead —
                # queued steers stay queued for the next run()'s head
                # fallback drain, never auto-injected by the interrupt.
                if self._turn_interrupted:
                    break
                pending = self.drain_steer((STEER,))
                if not pending:
                    break
                self._iteration_count = 0
                self._consecutive_length_stops = 0
                for item in pending:
                    yield self._inject_steer(item, followup=True)
                continue

            # Length-stop guard: the response was cut off by the output
            # token limit, so tool-call arguments may be truncated. Fail
            # the whole batch without executing it. OpenAI reports this as
            # "length"; Anthropic reports it as "max_tokens" (its
            # stop_reason is passed through verbatim by the client).
            if self._last_finish_reason in ("length", "max_tokens"):
                self._consecutive_length_stops += 1
                async for event in self._fail_truncated_tool_calls(last):
                    yield event
                if self._consecutive_length_stops >= _MAX_CONSECUTIVE_LENGTH_STOPS:
                    # Length-stop loop (LIM-32): the model keeps re-issuing a
                    # call that can never fit one response. Break the turn
                    # with actionable guidance instead of burning the whole
                    # iteration budget (mirrors pi's one-shot overflow-recovery
                    # guard: recoveries that keep failing stop early).
                    self.trace.log(
                        "error",
                        turn=self._turn_count,
                        iteration=self._iteration_count,
                        kind="length_stop_loop",
                        message=(
                            f"{self._consecutive_length_stops} consecutive responses "
                            "hit the output token limit; turn aborted."
                        ),
                    )
                    yield ErrorEvent(
                        message=(
                            f"Stopped after {self._consecutive_length_stops} consecutive "
                            "responses that hit the output token limit "
                            f"(max_tokens={self._effective_max_tokens()}): the model "
                            "keeps re-issuing a tool call that does not fit in one "
                            "response. Ask it to write the script to a file with the "
                            "write tool and run it via bash, split the task into "
                            "smaller steps, or switch to a model with a higher "
                            "output limit."
                        )
                    )
                    break
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

            # A batch that is about to execute proves the conversation is
            # making progress; reset the length-stop loop counter.
            self._consecutive_length_stops = 0
            for idx, tc in enumerate(last.tool_calls):
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
            async for event in self._execute_tool_calls(last):
                yield event
            await self._checkpoint()
            if self._turn_interrupted:
                # RFC LIM-53: batch results (real or placeholder) are all
                # recorded; end the turn instead of starting another call.
                break

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
        unexecuted: list[str] = []

        if self.config.tools.parallel and len(tool_calls) > 1:

            async def run_one(tc: dict[str, Any]) -> tuple[dict[str, Any], ToolResult]:
                result = await self._execute_tool_safe(
                    tc["id"], tc["function"]["name"], tc["function"]["arguments"]
                )
                return tc, result

            if self._interrupt_requested:
                # RFC LIM-53 pre-flight, symmetric with the sequential
                # branch: ESC landed between the end of the stream and the
                # start of the batch (cancel_active() had nothing to kill
                # yet) — no call starts, every call gets the placeholder.
                for tc in tool_calls:
                    result = ToolResult(success=False, error=_NOT_EXECUTED_MESSAGE)
                    results[tc["id"]] = result
                    unexecuted.append(tc["id"])
                    yield ToolResultEvent(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        result=result,
                        arguments=tc["function"]["arguments"],
                    )
            else:
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
                if self._interrupt_requested:
                    # RFC LIM-53: never-started calls get a placeholder.
                    # Only genuinely un-executed calls get it — started
                    # tools always record their real result, so history
                    # and disk cannot diverge.
                    result = ToolResult(success=False, error=_NOT_EXECUTED_MESSAGE)
                    unexecuted.append(tc["id"])
                else:
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
            content, attachments = self._gate_tool_attachments(result)
            self._history.record_result(tc["id"], content, attachments)

        # RFC LIM-53: an interrupt that arrived during the batch surfaces
        # now — every call already has its result recorded above (real for
        # started tools, placeholder for never-started ones).
        event = self._mark_interrupted(
            "tool_batch", unexecuted_tool_call_ids=unexecuted
        )
        if event is not None:
            yield event

    def _gate_tool_attachments(
        self, result: ToolResult
    ) -> tuple[str, list[Attachment] | None]:
        """Apply the vision gate to tool-result attachments.

        Mirrors the user-attachment policy (limbo.attachments): image
        payloads reach the model only when it supports vision; otherwise
        they degrade to a path-reference note in the text content so the
        model knows the file exists and where it is. Nothing is silently
        dropped.
        """
        content = result.output or result.error or ""
        images = [
            a for a in (result.attachments or []) if a.kind == "image"
        ]
        if not images:
            return content, None
        if self._vision:
            return content, result.attachments
        notes = "\n".join(
            f"[图片 {a.name} 位于 {a.path}；当前模型不支持图像输入，无法直接查看]"
            for a in images
        )
        content = f"{content}\n{notes}" if content else notes
        return content, None

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

    def _effective_max_tokens(self) -> int:
        """The output token cap in effect for the configured model."""
        return self.config.llm.max_tokens or resolve_model(
            self.config.llm.model
        ).max_tokens

    async def _fail_truncated_tool_calls(
        self, assistant: Message
    ) -> AsyncIterator[AgentEvent]:
        """Fail a whole batch without executing it (length-stop guard).

        A ``finish_reason == "length"`` response was cut off by the output
        token limit, so any tool call in it may carry truncated arguments
        that still parse. None of them are safe to execute; return an error
        result for each so the model can re-issue them (pi does the same).
        The message must name an alternative strategy: told only to
        "re-issue", the model re-sends the same oversized arguments and
        loops until the iteration fuse (LIM-32).
        """
        message = (
            "Tool call was not executed: the response hit the output token "
            f"limit (max_tokens={self._effective_max_tokens()}), so its "
            "arguments may be truncated. Do NOT re-issue the same call with "
            "the same arguments — it will be truncated again. Instead: "
            "write large scripts or content to a file with the write tool "
            "(in several smaller pieces if needed), then run it via bash; "
            "or split the work into multiple smaller tool calls."
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

    async def _stream_events(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        on_request: RequestHook | None,
    ) -> AsyncIterator[Any]:
        """Yield events from an LLM stream, interruptibly (RFC LIM-53).

        Sets ``self._stream_cut`` when the stream was cut short by an
        interrupt — callers use it (not the flag alone) to tell a cut
        stream apart from one that COMPLETED before the ESC landed (the
        flag may be set in the synchronous window between stream end and
        the tool batch; the completed response must be kept then). Both
        interrupt layers live here: the flag check between chunks
        (graceful, keeps partial content) and the cancellable pump task
        (covers stalled reads).
        """
        self._stream_cut = False
        stream = self.llm_client.chat(messages, tools=tools, on_request=on_request)
        try:
            while True:
                if self._interrupt_requested:
                    self._stream_cut = True
                    return
                try:
                    event = await self._anext_cancellable(stream)
                except StopAsyncIteration:
                    return
                except _StreamInterrupted:
                    self._stream_cut = True
                    return
                yield event
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
                    pass

    async def _call_llm(self) -> AsyncIterator[AgentEvent]:
        tool_definitions = self.registry.definitions()
        assistant_content = ""
        assistant_reasoning = ""
        assistant_signature: str | None = None
        tool_calls: list[dict[str, Any]] = []
        meta: CompletionMeta | None = None
        started_at = utc_now_iso()
        attempts = 0

        def _trace_request(body: dict[str, Any]) -> None:
            # Called once per HTTP attempt: retries share the iteration.
            nonlocal attempts
            attempts += 1
            self.trace.log(
                "llm_request",
                turn=self._turn_count,
                iteration=self._iteration_count,
                attempt=attempts,
                body=body,
            )

        async for event in self._stream_events(
            self.messages, tool_definitions, _trace_request
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

        if self._stream_cut:
            # RFC LIM-53: the stream was cut mid-response. The partial
            # response is appended content-only — partial tool-call
            # arguments may be truncated, and truncated thinking cannot be
            # replayed (its signature is incomplete), so both are dropped.
            # Dialect replays tolerate reasoning=None (Anthropic skips the
            # thinking block; Kimi replays reasoning_content as ""). A
            # stream that COMPLETED before the ESC landed takes the normal
            # path below; the batch pre-flight handles that interrupt.
            if assistant_content:
                self.messages.append(
                    Message(role="assistant", content=assistant_content)
                )
            event = self._mark_interrupted(
                "llm_stream",
                content_chars=len(assistant_content),
                content=assistant_content,
                reasoning=assistant_reasoning,
            )
            if event is not None:
                yield event
            return

        usage = meta.usage if meta else None
        totals = self.record_usage(usage)
        self._last_finish_reason = meta.finish_reason if meta else None
        if totals.total_tokens:
            yield UsageUpdate(total_tokens=self.session_total_tokens)
        self.trace.log(
            "llm_response",
            turn=self._turn_count,
            iteration=self._iteration_count,
            started_at=started_at,
            duration=meta.duration if meta else None,
            ttft=meta.ttft if meta else None,
            finish_reason=meta.finish_reason if meta else None,
            usage=usage,
            cached_tokens=totals.cached_tokens,
            content_chars=len(assistant_content),
            reasoning_chars=len(assistant_reasoning),
            content=assistant_content,
            reasoning=assistant_reasoning,
            tool_calls=[
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
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
        if totals.prompt_tokens is not None:
            self._prompt_estimator.record(totals.prompt_tokens, len(self.messages))

    # -- auto-compaction (LIM-14) --------------------------------------------

    def _estimated_next_prompt_tokens(self) -> int:
        """Estimated size of the *next* request's prompt (see llm.usage)."""
        return self._prompt_estimator.estimate_next(self.messages)

    async def _generate_summary(self, prompt: list[Message]) -> str:
        """One-shot summarization call on the current client (no tools)."""

        started_at = utc_now_iso()
        attempts = 0

        def _trace_request(body: dict[str, Any]) -> None:
            nonlocal attempts
            attempts += 1
            self.trace.log(
                "llm_request",
                turn=self._turn_count,
                iteration=self._iteration_count,
                kind="compaction",
                attempt=attempts,
                body=body,
            )

        text = ""
        meta: CompletionMeta | None = None
        async for event in self._stream_events(prompt, [], _trace_request):
            if isinstance(event, TextChunk):
                text += event.text
            elif isinstance(event, CompletionMeta):
                meta = event
        if self._interrupt_requested:
            # RFC LIM-53: the summary call was interrupted; compact()
            # reports it as an interruption, not a summary failure.
            raise _StreamInterrupted
        self.record_usage(meta.usage if meta else None)
        self.trace.log(
            "llm_response",
            turn=self._turn_count,
            iteration=self._iteration_count,
            kind="compaction",
            started_at=started_at,
            duration=meta.duration if meta else None,
            finish_reason=meta.finish_reason if meta else None,
            usage=meta.usage if meta else None,
            content_chars=len(text),
            content=text,
        )
        return text

    async def compact(self, trigger: str = "auto") -> AsyncIterator[AgentEvent]:
        """Compact history: summarize the old region, keep the recent tail.

        Shared by the loop-top auto trigger (``trigger="auto"``), the
        ``/compact`` slash command (``trigger="manual"``), and the recovery
        from a provider's context-overflow error (``trigger="overflow"``,
        followed by one resend of the request). On any failure
        the history is left untouched; an auto failure suppresses further
        auto attempts for the rest of the turn (the guard resets in
        :meth:`run`), while ``/compact`` can always be retried by hand.
        """
        cfg = self._compaction_config
        if trigger == "manual":
            # RFC LIM-53: /compact bypasses run(), so it resets the flag at
            # its own entry — an interrupt requested before it started must
            # not leak in. Auto-compact keeps a pending flag so ESC
            # mid-turn is honored at this consistency point.
            self._interrupt_requested = False
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
            elif trigger == "overflow":
                reason = "最近消息过大或历史太短，无法安全切分保留边界，无法通过压缩腾出上下文"
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
        compaction_started_at = utc_now_iso()
        start = time.monotonic()
        summary = ""
        error: str | None = None
        try:
            summary = await self._generate_summary(prompt)
            if not summary.strip():
                error = "empty response"
        except _StreamInterrupted:
            # RFC LIM-53: user interrupt — history is untouched (the
            # rewrite below never runs); report and stop.
            event = self._mark_interrupted("compact")
            if event is not None:
                yield event
            yield CompactionEvent(
                compacted=False,
                trigger=trigger,
                before_tokens=before,
                reason="已被用户打断（ESC）",
            )
            return
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
        summarized_unsaved = self.messages[self._session_saved_count : split]
        # Kept messages not yet on disk (e.g. this turn's user input): the
        # append path must still write them after the compaction.
        unsaved = min(len(self.messages) - self._session_saved_count, len(kept))
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
        if is_summary_message(kept[0]):
            # The boundary is an earlier summary, which lives only in memory
            # on the append path: the record could not be resolved on load,
            # so this one compaction falls back to a full rewrite.
            self._session_full_rewrite_needed = True
        else:
            # Non-destructive compaction (pi-durable): append the record;
            # the summarized messages stay in the journal and loading picks
            # the active range from the record's boundary.
            self._pending_summarized.extend(summarized_unsaved)
            self._pending_compactions.append(record)
            self._session_saved_count = len(self.messages) - max(unsaved, 0)
        self._previous_summary = summary.strip()
        # Usage figures predate the rewrite; fall back to full estimation
        # until the next response reports real prompt tokens.
        self._prompt_estimator.reset(len(self.messages))
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
            started_at=compaction_started_at,
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
        # The summary call is billed too: refresh the counter.
        if self.session_total_tokens:
            yield UsageUpdate(total_tokens=self.session_total_tokens)

    def _save_session_sync(self) -> None:
        """Persist the session: incremental append by default, full rewrite
        when the on-disk state changes shape.

        Full rewrite happens when the file does not exist yet (first save),
        when compaction reshaped the history, when a resumed session's
        repaired history must be re-synced, or when the meta record changed
        (title, code_mode, goal, ...) without having been patched in via
        :meth:`persist_meta` — the meta line is only written by a full
        rewrite. The meta snapshot comparison replaces ad-hoc dirty flags,
        so future meta fields are covered automatically. Otherwise only the
        messages appended since the last save are written — long sessions
        no longer rewrite megabytes every turn.
        """
        # Keep the presentation mode in sync before snapshotting the meta.
        self._meta.code_mode = self.code_mode
        if not self._meta.title:
            self._meta.title = derive_title(self.messages)
        meta_snapshot = self._meta_snapshot()
        with self._session_io_lock:
            self._write_session_locked(meta_snapshot)

    def _meta_snapshot(self) -> str:
        """The meta content compared against what is on disk.

        updated_at is stamped by save_session (full rewrite) and refreshed
        in-memory on the append path; it is excluded so the comparison
        sees real meta content, not the always-moving timestamp.
        """
        return json.dumps(
            self._meta.model_dump(exclude={"path", "updated_at"}),
            ensure_ascii=False,
            sort_keys=True,
        )

    def _write_session_locked(self, meta_snapshot: str) -> None:
        if self._session_full_rewrite_needed or not self._session_file.exists():
            save_session(
                self._session_file,
                self._meta,
                self.messages,
                self._compactions,
            )
            self._session_saved_count = len(self.messages)
            self._session_saved_meta = meta_snapshot
            self._session_saved_title = self._meta.title
            self._session_full_rewrite_needed = False
            self._pending_compactions.clear()
            self._pending_summarized.clear()
            # The rewrite dropped the queue journal: re-journal what is
            # still queued (everything else has left the queue).
            self._pending_dequeued.clear()
            queued = [queued_record(_queue_record(i)) for i in self._steer_queue]
            if queued:
                try:
                    append_session_records(self._session_file, queued)
                except OSError:
                    # The file holds no queue journal: rebuild it next save.
                    self._session_full_rewrite_needed = True
                    raise
            return
        try:
            if self._meta.title != self._session_saved_title:
                # Listing reads only the meta line: rewrite it in place
                # (every other record is kept).
                rewrite_meta_line(self._session_file, self._meta)
                self._session_saved_title = self._meta.title
                self._session_saved_meta = meta_snapshot
            elif meta_snapshot != self._session_saved_meta:
                append_session_records(
                    self._session_file, [meta_patch_record(self._meta)]
                )
                self._session_saved_meta = meta_snapshot
            records = [compaction_record(c) for c in self._pending_compactions]
            new_messages = self.messages[self._session_saved_count :]
            if self._pending_summarized:
                append_session_messages(self._session_file, self._pending_summarized)
                self._pending_summarized.clear()
            if records:
                append_session_records(self._session_file, records)
                self._pending_compactions.clear()
            if new_messages:
                append_session_messages(self._session_file, new_messages)
                self._session_saved_count = len(self.messages)
            if self._pending_dequeued:
                # After the messages: an injected item must never vanish
                # from the queue before its message is on disk.
                append_session_records(
                    self._session_file, [dequeued_record(self._pending_dequeued)]
                )
                self._pending_dequeued.clear()
        except OSError:
            # Partial or failed append: some records may already be on disk,
            # and re-appending next time would duplicate them. Force a full
            # rewrite on the next save, which rebuilds the file from memory.
            self._session_full_rewrite_needed = True
            raise
        # Keep the in-memory timestamp fresh even though the meta line
        # is not rewritten (export headers read it).
        self._meta.updated_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )

    async def _save_session(self) -> None:
        # Trace and session are separate files written independently; the
        # flush barrier belongs here (turn end), so by the time run()
        # returns, every trace record is on disk. Both run off the event
        # loop so the UI never blocks on either file.
        def _sync() -> None:
            self.trace.flush()
            self._save_session_sync()

        await asyncio.to_thread(_sync)

    async def _checkpoint(self) -> None:
        """Persist history at a mid-turn consistency point.

        Durable progress (pi-durable's "only committed state is visible"):
        the loop calls this before each LLM request, after each response,
        and after each tool batch, so a killed process loses at most the
        in-flight step instead of the whole turn. Cheap in steady state —
        an append of the new messages only. A failure never breaks the
        turn: it is traced, warned once per turn, and retried at the next
        checkpoint (a failed append already forces a full rewrite).
        """
        try:
            await asyncio.to_thread(self._save_session_sync)
        except Exception as e:  # noqa: BLE001
            self._report_save_error("checkpoint", e)

    def _report_save_error(self, phase: str, error: Exception) -> None:
        """Trace a mid-turn persistence failure; warn at most once per turn."""
        self.trace.log(
            "session_save_error",
            turn=self._turn_count,
            iteration=self._iteration_count,
            phase=phase,
            error=str(error),
        )
        if not self._checkpoint_failed_this_turn:
            self._checkpoint_failed_this_turn = True
            warnings.warn(f"Failed to save session: {error}", stacklevel=3)

    def persist_meta(self) -> None:
        """Persist a meta change now, as an appended ``meta_patch`` line.

        Used for state that must survive a crash before the next turn-end
        save (the goal state written by ``TurnPump``). Appending avoids a
        full rewrite; loading merges patches over the meta line. Before the
        first save there is no file yet — the first save writes the meta.
        """
        if not self._session_file.exists():
            return
        self._meta.code_mode = self.code_mode
        snapshot = self._meta_snapshot()
        with self._session_io_lock:
            if snapshot == self._session_saved_meta:
                return
            append_session_records(
                self._session_file, [meta_patch_record(self._meta)]
            )
            self._session_saved_meta = snapshot


def _queue_record(item: SteerItem) -> QueuedMessageRecord:
    return QueuedMessageRecord(
        id=item.id, text=item.text, attachments=item.attachments, mode=item.mode
    )
