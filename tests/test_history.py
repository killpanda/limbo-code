"""Tests for the ToolHistory seam (limbo.history)."""

from __future__ import annotations

from limbo.history import ToolHistory, repair
from limbo.models import Message


def assistant_with_calls(*ids: str) -> Message:
    return Message(
        role="assistant",
        content=None,
        tool_calls=[
            {"id": i, "type": "function", "function": {"name": "read", "arguments": "{}"}}
            for i in ids
        ],
    )


def test_record_result_appends_when_missing():
    messages = [Message(role="user", content="hi")]
    history = ToolHistory(messages)

    history.record_result("c1", "output")

    assert messages[-1].role == "tool"
    assert messages[-1].tool_call_id == "c1"
    assert messages[-1].content == "output"


def test_record_result_replaces_existing_in_place():
    placeholder = Message(role="tool", content="pending", tool_call_id="c1")
    messages = [assistant_with_calls("c1"), placeholder]
    history = ToolHistory(messages)

    history.record_result("c1", "real output")

    tool_msgs = [m for m in messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].content == "real output"


def test_install_placeholders_tracks_pending_ids():
    messages = [assistant_with_calls("c1", "c2")]
    history = ToolHistory(messages)

    history.install_placeholders(["c1", "c2"])

    assert history.pending_ids == {"c1", "c2"}
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2"]
    # Recording a result clears its pending state.
    history.record_result("c1", "done")
    assert history.pending_ids == {"c2"}


def test_record_error_marks_crashed_call_and_siblings():
    messages = [assistant_with_calls("c1", "c2", "c3")]
    history = ToolHistory(messages)

    history.record_error(messages[0], 1, "c2", "Tool error: boom")

    results = {m.tool_call_id: m.content for m in messages if m.role == "tool"}
    assert results == {
        "c2": "Tool error: boom",
        "c3": "Action not executed: earlier tool failed.",
    }


def test_record_rejection_marks_pending_and_later_calls():
    messages = [
        assistant_with_calls("c1", "c2", "c3"),
        Message(role="tool", content="ok", tool_call_id="c1"),
        Message(role="tool", content="pending", tool_call_id="c2"),
        Message(role="tool", content="pending", tool_call_id="c3"),
    ]
    history = ToolHistory(messages, pending_ids={"c2", "c3"})

    history.record_rejection("c2", reason="nope")

    results = {m.tool_call_id: m.content for m in messages if m.role == "tool"}
    assert results["c1"] == "ok"
    assert results["c2"] == "nope"
    assert results["c3"] == "Action not executed: earlier tool was rejected."
    assert history.pending_ids == set()


def test_record_rejection_fallback_without_owner_assistant():
    messages = [Message(role="tool", content="pending", tool_call_id="c9")]
    history = ToolHistory(messages)

    history.record_rejection("c9")

    assert messages[0].content == "Action rejected."


def test_repair_drops_system_and_fixes_dangling_tool_calls():
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        assistant_with_calls("c1"),
    ]

    restored = repair(messages)

    assert [m.role for m in restored] == ["user", "assistant", "tool"]
    assert restored[-1].tool_call_id == "c1"
    assert "interrupted" in (restored[-1].content or "")


def test_repair_keeps_complete_histories_untouched():
    messages = [
        Message(role="user", content="hi"),
        assistant_with_calls("c1"),
        Message(role="tool", content="result", tool_call_id="c1"),
        Message(role="assistant", content="done"),
    ]

    restored = repair(messages)

    assert [m.model_dump() for m in restored] == [m.model_dump() for m in messages]
