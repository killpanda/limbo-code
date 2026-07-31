"""Headless (non-interactive) execution mode.

``limbo exec`` runs exactly one agent turn — one prompt in, events out —
without the TUI. It exists for batch drivers (e.g. a SWE-bench runner)
that wrap each instance in a subprocess:

- The prompt comes from ``--prompt-file``/``--prompt``; there is no stdin
  interaction and the steer queue stays empty by construction.
- Assistant text goes to stdout; tool calls, results, compaction, and
  errors go to stderr as one-line progress records, so the driver can
  keep a readable log per instance.
- Wall-clock timeouts and patch collection are the *driver's* job: it
  kills the subprocess on timeout and runs ``git diff`` in the workdir
  afterwards. This module deliberately adds neither.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from limbo.agent import (
    Agent,
    AgentEvent,
    CompactionEvent,
    ErrorEvent,
    SteerEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallRequest,
    ToolResultEvent,
    UsageUpdate,
)
from limbo.config import Config
from limbo.llm.factory import create_llm_client

MAX_LOG_ARG_CHARS = 200


def _log(message: str) -> None:
    """Emit one progress line on stderr (stdout stays assistant-text only)."""
    elapsed = time.strftime("%H:%M:%S")
    print(f"[{elapsed}] {message}", file=sys.stderr, flush=True)


def _summarize_arguments(arguments: dict[str, object]) -> str:
    text = json.dumps(arguments, ensure_ascii=False, default=str)
    if len(text) > MAX_LOG_ARG_CHARS:
        text = text[: MAX_LOG_ARG_CHARS - 1] + "…"
    return text


async def run_headless(
    prompt: str,
    *,
    config: Config,
    workdir: Path,
    session_dir: Path | None = None,
) -> int:
    """Run one agent turn to completion. Returns a process exit code."""
    agent = Agent(
        config=config,
        llm_client=create_llm_client(config),
        workdir=workdir,
        session_dir=session_dir,
    )
    _log(f"exec start: workdir={workdir} model={config.llm.model}")

    had_error = False
    total_tokens = 0
    async for event in agent.run(prompt):
        had_error, total_tokens = _handle_event(event, had_error, total_tokens)

    _log(f"exec end: total_tokens={total_tokens} status={'error' if had_error else 'ok'}")
    return 1 if had_error else 0


def _handle_event(
    event: AgentEvent, had_error: bool, total_tokens: int
) -> tuple[bool, int]:
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ThinkingDelta):
        pass  # Reasoning is recorded in the trace; keep stderr terse.
    elif isinstance(event, ToolCallRequest):
        _log(f"tool call: {event.name} {_summarize_arguments(event.arguments)}")
    elif isinstance(event, ToolResultEvent):
        if event.result.success:
            output = event.result.output or ""
            _log(f"tool ok: {event.name} ({len(output)} chars)")
        else:
            _log(f"tool fail: {event.name}: {event.result.error or 'unknown'}")
    elif isinstance(event, ErrorEvent):
        _log(f"ERROR: {event.message}")
        had_error = True
    elif isinstance(event, CompactionEvent):
        if event.compacted:
            _log(
                f"compacted ({event.trigger}): ~{event.before_tokens} -> "
                f"{event.after_estimate} tokens"
            )
        elif event.reason:
            _log(f"compaction skipped: {event.reason}")
        if event.warning:
            _log(f"compaction warning: {event.warning}")
    elif isinstance(event, UsageUpdate):
        total_tokens = event.total_tokens
    elif isinstance(event, SteerEvent):
        _log(f"unexpected steer in headless mode: {event.text!r}")
    return had_error, total_tokens
