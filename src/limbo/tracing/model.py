"""Assemble flat trace records into the viewer's session tree.

Mapping (Langfuse vocabulary): the session is a Session, each user turn a
Trace, each LLM call a Generation, each tool execution a Span, and every
other record an Event on its turn.

Turns are segmented by record order, not by the ``turn`` field: a resumed
session appends to the same trace file with a fresh agent whose turn
counter restarts at 1. A turn opens at a top-level ``user_message`` (one
without ``steer``/``followup``, which are mid-turn injections) and closes
at ``turn_end``. Work outside any turn (``/compact`` at idle) opens an
implicit turn with no input.

Timing: schema-2 records carry ``started_at``; older ones fall back to
``ts - duration`` (``ts`` is written when the record is logged, i.e. at
the end of the timed operation).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from limbo.llm.usage import normalize_usage
from limbo.tracing.index import IndexedRecord

_ERROR_TYPES = ("llm_error", "error", "compaction_error", "session_save_error")


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _start_of(summary: dict[str, Any]) -> str | None:
    """Start time of a timed record (see module docstring)."""
    started = summary.get("started_at")
    if isinstance(started, str):
        return started
    end = _parse_ts(summary.get("ts"))
    duration = summary.get("duration")
    if end is None or not isinstance(duration, (int, float)):
        return summary.get("ts")
    return (end - timedelta(seconds=duration)).isoformat(timespec="milliseconds")


def _tokens(usage: Any) -> dict[str, int | None]:
    totals = normalize_usage(usage if isinstance(usage, dict) else None)
    completion = None
    if isinstance(usage, dict):
        for key in ("completion_tokens", "output_tokens"):
            if isinstance(usage.get(key), int):
                completion = usage[key]
                break
    if completion is None and totals.total_tokens is not None and totals.prompt_tokens:
        completion = totals.total_tokens - totals.prompt_tokens
    return {
        "input": totals.prompt_tokens,
        "output": completion,
        "cached": totals.cached_tokens,
        "total": totals.total_tokens,
    }


def _add_tokens(into: dict[str, int], tokens: dict[str, int | None]) -> None:
    for key, value in tokens.items():
        if value:
            into[key] = into.get(key, 0) + value


def _empty_totals() -> dict[str, Any]:
    return {
        "tokens": {},
        "generations": 0,
        "tools": 0,
        "errors": 0,
        "llm_seconds": 0.0,
        "tool_seconds": 0.0,
    }


class _Builder:
    def __init__(self) -> None:
        self.session: dict[str, Any] = {}
        self.turns: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []  # session-level (outside turns)
        self.totals = _empty_totals()
        self._open: dict[str, Any] | None = None
        self._generations: dict[tuple[Any, Any], dict[str, Any]] = {}
        self._tools: dict[Any, dict[str, Any]] = {}

    # -- turns ---------------------------------------------------------------

    def _open_turn(self, ts: Any, input_preview: str | None, offset: int | None) -> dict[str, Any]:
        turn = {
            "index": len(self.turns) + 1,
            "input": input_preview,
            "input_offset": offset,
            "start": ts,
            "end": None,
            "duration": None,
            "status": "running",
            "spans": [],
            "totals": _empty_totals(),
        }
        self.turns.append(turn)
        self._open = turn
        self._generations.clear()
        self._tools.clear()
        return turn

    def _turn(self, summary: dict[str, Any]) -> dict[str, Any]:
        if self._open is None:
            self._open_turn(summary.get("ts"), None, None)
        assert self._open is not None
        return self._open

    def _event(self, summary: dict[str, Any], offset: int) -> None:
        event = {
            "kind": "event",
            "type": summary.get("type"),
            "ts": summary.get("ts"),
            "offset": offset,
            "summary": {k: v for k, v in summary.items() if k not in ("ts", "type")},
        }
        if self._open is not None:
            self._open["spans"].append(event)
        else:
            self.events.append(event)

    def _error(self, turn: dict[str, Any] | None) -> None:
        self.totals["errors"] += 1
        if turn is not None:
            turn["totals"]["errors"] += 1

    # -- records -------------------------------------------------------------

    def add(self, item: IndexedRecord) -> None:
        summary, offset = item.summary, item.offset
        kind = summary.get("type")
        if kind == "session_start":
            if self._open is not None:
                # A resume after a crash: the previous run never logged
                # turn_end, and nothing after this belongs to that turn.
                self._open["status"] = "unfinished"
                self._fail_pending(summary, "unfinished")
                self._open = None
            if not self.session:
                self.session = {k: v for k, v in summary.items() if k != "type"}
                self.session["offset"] = offset
            self._event(summary, offset)
        elif kind == "user_message" and not summary.get("steer") and not summary.get("followup"):
            if self._open is not None:  # previous turn never logged turn_end
                self._open["status"] = "unfinished"
            self._open_turn(summary.get("ts"), summary.get("content_preview"), offset)
        elif kind == "llm_request":
            self._request(summary, offset)
        elif kind == "llm_response":
            self._response(summary, offset)
        elif kind == "tool_call":
            self._tool_call(summary, offset)
        elif kind == "tool_result":
            self._tool_result(summary, offset)
        elif kind == "turn_end":
            self._turn_end(summary)
        else:
            if kind in _ERROR_TYPES:
                self._error(self._open)
                self._fail_pending(summary, "error")
            elif kind == "turn_interrupted":
                self._fail_pending(summary, "interrupted")
            elif kind == "compaction" and self._open is not None:
                self._compaction(summary, offset)
                return
            self._event(summary, offset)

    def _generation_key(self, summary: dict[str, Any]) -> tuple[Any, Any]:
        return (summary.get("iteration"), summary.get("kind"))

    def _request(self, summary: dict[str, Any], offset: int) -> None:
        turn = self._turn(summary)
        key = self._generation_key(summary)
        generation = self._generations.get(key)
        if generation is None or generation["status"] != "running":
            generation = {
                "kind": "generation",
                "id": f"g{offset}",
                "name": "compaction" if summary.get("kind") == "compaction" else "llm",
                "iteration": summary.get("iteration"),
                "model": summary.get("model"),
                "message_count": summary.get("message_count"),
                "start": summary.get("ts"),
                "end": None,
                "duration": None,
                "status": "running",
                "attempts": [],
                "response_offset": None,
                "tokens": {},
            }
            self._generations[key] = generation
            turn["spans"].append(generation)
        generation["attempts"].append(offset)

    def _response(self, summary: dict[str, Any], offset: int) -> None:
        turn = self._turn(summary)
        key = self._generation_key(summary)
        generation = self._generations.get(key)
        if generation is None:  # request record lost (old/partial trace)
            generation = {
                "kind": "generation",
                "id": f"g{offset}",
                "name": "compaction" if summary.get("kind") == "compaction" else "llm",
                "iteration": summary.get("iteration"),
                "attempts": [],
            }
            turn["spans"].append(generation)
        tokens = _tokens(summary.get("usage"))
        duration = summary.get("duration")
        generation.update(
            start=_start_of(summary) or generation.get("start"),
            end=summary.get("ts"),
            duration=duration,
            ttft=summary.get("ttft"),
            finish_reason=summary.get("finish_reason"),
            tokens=tokens,
            tool_calls=summary.get("tool_calls") or [],
            output_preview=summary.get("content_preview"),
            status="ok",
            response_offset=offset,
        )
        self._generations[key] = generation
        for totals in (self.totals, turn["totals"]):
            totals["generations"] += 1
            _add_tokens(totals["tokens"], tokens)
            if isinstance(duration, (int, float)):
                totals["llm_seconds"] += duration

    def _tool_call(self, summary: dict[str, Any], offset: int) -> None:
        turn = self._turn(summary)
        span = {
            "kind": "tool",
            "id": f"t{offset}",
            "call_id": summary.get("id"),
            "name": summary.get("name"),
            "iteration": summary.get("iteration"),
            "start": summary.get("ts"),
            "end": None,
            "duration": None,
            "status": "running",
            "call_offset": offset,
            "result_offset": None,
            "arguments_preview": summary.get("arguments_preview"),
            # Code Mode sub-call: nested under its run_code call.
            "parent_call_id": summary.get("parent_call_id"),
        }
        self._tools[summary.get("id")] = span
        turn["spans"].append(span)

    def _tool_result(self, summary: dict[str, Any], offset: int) -> None:
        turn = self._turn(summary)
        span = self._tools.pop(summary.get("id"), None)
        if span is None:
            span = {
                "kind": "tool",
                "id": f"t{offset}",
                "name": summary.get("name"),
                "call_offset": None,
            }
            turn["spans"].append(span)
        duration = summary.get("duration")
        failed = summary.get("success") is False
        span.update(
            start=_start_of(summary),
            end=summary.get("ts"),
            duration=duration,
            status="error" if failed else "ok",
            result_offset=offset,
            output_preview=summary.get("output_preview"),
            error=summary.get("error"),
        )
        # Sub-call time is already inside its run_code parent's duration.
        sub_call = summary.get("parent_call_id") is not None
        for totals in (self.totals, turn["totals"]):
            if sub_call:
                break
            totals["tools"] += 1
            if isinstance(duration, (int, float)):
                totals["tool_seconds"] += duration
        if failed:
            self._error(turn)

    def _compaction(self, summary: dict[str, Any], offset: int) -> None:
        assert self._open is not None
        self._open["spans"].append(
            {
                "kind": "compaction",
                "id": f"c{offset}",
                "name": "compaction",
                "start": _start_of(summary),
                "end": summary.get("ts"),
                "duration": summary.get("duration"),
                "status": "ok",
                "offset": offset,
                "summary": {k: v for k, v in summary.items() if k not in ("ts", "type")},
            }
        )

    def _fail_pending(self, summary: dict[str, Any], status: str) -> None:
        """Close running spans of the open turn at an error/interrupt."""
        if self._open is None:
            return
        for span in self._open["spans"]:
            if span.get("status") == "running":
                span["status"] = status
                span["end"] = summary.get("ts")

    def _turn_end(self, summary: dict[str, Any]) -> None:
        turn = self._turn(summary)
        turn["end"] = summary.get("ts")
        turn["duration"] = summary.get("duration")
        if summary.get("started_at"):
            turn["start"] = summary["started_at"]
        status = summary.get("status") or "completed"
        turn["status"] = status
        self._fail_pending(summary, "interrupted" if status == "interrupted" else "unfinished")
        self._open = None


def build_tree(records: list[IndexedRecord]) -> dict[str, Any]:
    """The whole session as JSON-ready dicts (see module docstring)."""
    builder = _Builder()
    for item in records:
        builder.add(item)
    return {
        "session": builder.session,
        "events": builder.events,
        "turns": builder.turns,
        "totals": builder.totals,
        "record_count": len(records),
        "live": builder._open is not None,
    }


def _request_messages(body: dict[str, Any]) -> list[Any]:
    messages = body.get("messages", body.get("input"))
    return messages if isinstance(messages, list) else []


def request_diff(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """Messages new in ``current``'s request body since ``previous``'s.

    Every agent request resends the whole history, so the interesting part
    is the tail past the longest common prefix. After compaction the prefix
    collapses and the whole (rewritten) history shows as new — correct.
    """
    messages = _request_messages(current.get("body") or {})
    if previous is None:
        return {"prefix": 0, "total": len(messages), "messages": messages}
    old = _request_messages(previous.get("body") or {})
    prefix = 0
    for a, b in zip(old, messages):
        if a != b:
            break
        prefix += 1
    return {"prefix": prefix, "total": len(messages), "messages": messages[prefix:]}
