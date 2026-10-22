"""Tests for the session tree assembly (limbo.tracing.model)."""

from __future__ import annotations

from limbo.tracing.index import IndexedRecord, summarize
from limbo.tracing.model import build_tree


def records(*raw: dict) -> list[IndexedRecord]:
    return [IndexedRecord(i * 100, summarize(r)) for i, r in enumerate(raw)]


USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 50,
    "total_tokens": 1050,
    "prompt_cache_hit_tokens": 800,
}


def full_turn(turn: int = 1) -> list[dict]:
    return [
        {
            "type": "user_message",
            "ts": "2026-01-01T00:00:00.000+00:00",
            "turn": turn,
            "content": "fix it",
        },
        {
            "type": "llm_request",
            "ts": "2026-01-01T00:00:00.100+00:00",
            "turn": turn,
            "iteration": 1,
            "attempt": 1,
            "body": {"model": "m", "messages": [1, 2]},
        },
        {
            "type": "llm_response",
            "ts": "2026-01-01T00:00:01.100+00:00",
            "turn": turn,
            "iteration": 1,
            "started_at": "2026-01-01T00:00:00.100+00:00",
            "duration": 1.0,
            "ttft": 0.2,
            "usage": USAGE,
            "finish_reason": "tool_calls",
            "content": "reading",
            "tool_calls": [{"id": "c1", "name": "read", "arguments": {}}],
        },
        {
            "type": "tool_call",
            "ts": "2026-01-01T00:00:01.110+00:00",
            "turn": turn,
            "iteration": 1,
            "id": "c1",
            "name": "read",
            "arguments": {"path": "a.py"},
        },
        {
            "type": "tool_result",
            "ts": "2026-01-01T00:00:01.200+00:00",
            "turn": turn,
            "iteration": 1,
            "id": "c1",
            "name": "read",
            "success": False,
            "error": "File not found",
            "duration": 0.09,
            "started_at": "2026-01-01T00:00:01.110+00:00",
        },
        {
            "type": "llm_request",
            "ts": "2026-01-01T00:00:01.300+00:00",
            "turn": turn,
            "iteration": 2,
            "body": {"model": "m", "messages": [1, 2, 3, 4]},
        },
        {
            "type": "llm_response",
            "ts": "2026-01-01T00:00:02.300+00:00",
            "turn": turn,
            "iteration": 2,
            "duration": 1.0,
            "usage": USAGE,
            "finish_reason": "stop",
        },
        {
            "type": "turn_end",
            "ts": "2026-01-01T00:00:02.310+00:00",
            "turn": turn,
            "duration": 2.31,
            "iterations": 2,
            "status": "completed",
        },
    ]


def test_full_turn_builds_generations_and_tool_spans():
    tree = build_tree(
        records({"type": "session_start", "ts": "x", "session_id": "s"}, *full_turn())
    )
    assert tree["session"]["session_id"] == "s"
    assert tree["live"] is False
    (turn,) = tree["turns"]
    assert turn["input"] == "fix it"
    assert turn["status"] == "completed"
    kinds = [s["kind"] for s in turn["spans"]]
    assert kinds == ["generation", "tool", "generation"]

    gen = turn["spans"][0]
    assert gen["model"] == "m" and gen["message_count"] == 2
    assert gen["start"] == "2026-01-01T00:00:00.100+00:00"
    assert gen["tokens"] == {"input": 1000, "output": 50, "cached": 800, "total": 1050}
    assert gen["tool_calls"] == [{"id": "c1", "name": "read"}]
    assert gen["status"] == "ok"

    tool = turn["spans"][1]
    assert tool["status"] == "error" and tool["error"] == "File not found"
    assert tool["call_offset"] is not None and tool["result_offset"] is not None

    # Old-schema response without started_at: start = ts - duration.
    assert turn["spans"][2]["start"] == "2026-01-01T00:00:01.300+00:00"

    totals = tree["totals"]
    assert totals["generations"] == 2 and totals["tools"] == 1 and totals["errors"] == 1
    assert totals["tokens"]["input"] == 2000
    assert turn["totals"]["llm_seconds"] == 2.0


def test_resume_after_crash_closes_the_open_turn():
    crashed = full_turn()[:4]  # no turn_end: the process died mid-tool
    tree = build_tree(records(*crashed, {"type": "session_start", "ts": "x"}))
    (turn,) = tree["turns"]
    assert turn["status"] == "unfinished"
    assert turn["spans"][-1]["status"] == "unfinished"
    assert tree["live"] is False
    assert tree["events"][-1]["type"] == "session_start"


