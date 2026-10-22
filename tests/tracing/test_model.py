"""Tests for the session tree assembly (limbo.tracing.model)."""

from __future__ import annotations

from limbo.tracing.index import IndexedRecord, summarize
from limbo.tracing.model import build_tree, request_diff


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


def test_retries_group_into_one_generation():
    tree = build_tree(
        records(
            {"type": "user_message", "ts": "t", "content": "q"},
            {"type": "llm_request", "ts": "t", "iteration": 1, "attempt": 1, "body": {}},
            {"type": "llm_request", "ts": "t", "iteration": 1, "attempt": 2, "body": {}},
            {"type": "llm_response", "ts": "t", "iteration": 1, "duration": 1.0},
        )
    )
    (gen,) = tree["turns"][0]["spans"]
    assert len(gen["attempts"]) == 2


def test_resumed_session_segments_turns_by_order():
    # A resumed agent restarts its turn counter at 1 in the same file.
    tree = build_tree(records(*full_turn(1), {"type": "session_start", "ts": "x"}, *full_turn(1)))
    assert [t["index"] for t in tree["turns"]] == [1, 2]
    assert all(len(t["spans"]) == 3 for t in tree["turns"])


def test_resume_after_crash_closes_the_open_turn():
    crashed = full_turn()[:4]  # no turn_end: the process died mid-tool
    tree = build_tree(records(*crashed, {"type": "session_start", "ts": "x"}))
    (turn,) = tree["turns"]
    assert turn["status"] == "unfinished"
    assert turn["spans"][-1]["status"] == "unfinished"
    assert tree["live"] is False
    assert tree["events"][-1]["type"] == "session_start"


def test_open_turn_is_live_and_running():
    raw = full_turn()[:4]  # stops after tool_call
    tree = build_tree(records(*raw))
    assert tree["live"] is True
    turn = tree["turns"][0]
    assert turn["status"] == "running"
    assert turn["spans"][-1]["status"] == "running"


def test_interrupt_and_llm_error_close_running_spans():
    tree = build_tree(
        records(
            {"type": "user_message", "ts": "t0", "content": "q"},
            {"type": "llm_request", "ts": "t1", "iteration": 1, "body": {}},
            {"type": "llm_error", "ts": "t2", "iteration": 1, "error": "boom"},
            {"type": "turn_end", "ts": "t3", "status": "error", "duration": 1},
        )
    )
    turn = tree["turns"][0]
    assert turn["status"] == "error"
    assert turn["spans"][0]["status"] == "error"
    assert turn["spans"][1]["kind"] == "event" and turn["spans"][1]["type"] == "llm_error"
    assert tree["totals"]["errors"] == 1

    tree = build_tree(
        records(
            {"type": "user_message", "ts": "t0", "content": "q"},
            {"type": "llm_request", "ts": "t1", "iteration": 1, "body": {}},
            {"type": "turn_interrupted", "ts": "t2", "iteration": 1, "phase": "llm_stream"},
            {"type": "turn_end", "ts": "t3", "status": "interrupted"},
        )
    )
    assert tree["turns"][0]["status"] == "interrupted"
    assert tree["turns"][0]["spans"][0]["status"] == "interrupted"


def test_steer_messages_stay_inside_the_turn():
    tree = build_tree(
        records(
            {"type": "user_message", "ts": "t", "content": "q"},
            {
                "type": "user_message",
                "ts": "t",
                "content": "also",
                "steer": True,
                "followup": False,
            },
            {"type": "turn_end", "ts": "t", "status": "completed"},
        )
    )
    assert len(tree["turns"]) == 1
    assert tree["turns"][0]["spans"][0]["type"] == "user_message"


def test_idle_compaction_opens_implicit_turn():
    tree = build_tree(
        records(
            {"type": "llm_request", "ts": "t", "iteration": 0, "kind": "compaction", "body": {}},
            {
                "type": "llm_response",
                "ts": "t",
                "iteration": 0,
                "kind": "compaction",
                "duration": 1,
            },
            {"type": "compaction", "ts": "t", "duration": 1.2, "tokens_before": 9000},
        )
    )
    (turn,) = tree["turns"]
    assert turn["input"] is None
    assert [s["kind"] for s in turn["spans"]] == ["generation", "compaction"]
    assert turn["spans"][0]["name"] == "compaction"


def test_request_diff_returns_new_tail():
    old = {"body": {"messages": [{"role": "system"}, {"role": "user", "content": "a"}]}}
    new = {
        "body": {
            "messages": [
                {"role": "system"},
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
    }
    diff = request_diff(new, old)
    assert diff == {"prefix": 2, "total": 3, "messages": [{"role": "assistant", "content": "b"}]}
    assert request_diff(new, None)["prefix"] == 0
    # Responses API bodies carry `input` instead of `messages`.
    assert request_diff({"body": {"input": [1, 2]}}, {"body": {"input": [1]}})["messages"] == [2]
