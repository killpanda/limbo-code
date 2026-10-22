"""Tests for the ToolHistory seam (limbo.history)."""

from __future__ import annotations

from limbo.history import (
    INTERRUPTED_CONTENT,
    NOT_STARTED_CONTENT,
    PARTIAL_CONTENT,
    recover,
    repair,
)
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


def _call(call_id: str, name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": {}},
    }


def test_recover_classifies_dangling_calls_by_intent_and_policy():
    messages = [
        Message(role="user", content="go"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                _call("r1", "read"),
                _call("b1", "bash"),
                _call("w1", "write"),
            ],
        ),
    ]

    recovery = recover(
        messages, started={"r1", "b1"}, is_safe=lambda name: name == "read"
    )

    contents = {
        m.tool_call_id: m.content for m in recovery.messages if m.role == "tool"
    }
    assert contents == {
        # Safe: queued for re-run; the placeholder stands until it runs.
        "r1": INTERRUPTED_CONTENT,
        # Intent committed, side effects possible.
        "b1": PARTIAL_CONTENT,
        # No intent: the tool never started.
        "w1": NOT_STARTED_CONTENT,
    }
    assert [tc["id"] for tc in recovery.rerun] == ["r1"]


