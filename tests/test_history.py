"""Tests for the ToolHistory seam (limbo.history)."""

from __future__ import annotations

from limbo.history import (
    INTERRUPTED_CONTENT,
    NOT_STARTED_CONTENT,
    PARTIAL_CONTENT,
    ToolHistory,
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


def test_record_result_stores_attachments():
    from limbo.models import Attachment

    messages = [Message(role="user", content="hi")]
    history = ToolHistory(messages)
    attachment = Attachment(kind="image", name="p.png", path="/tmp/p.png", mime="image/png")

    history.record_result("c1", "Read image file [image/png]", [attachment])

    assert messages[-1].role == "tool"
    assert messages[-1].attachments == [attachment]


def test_record_result_replaces_attachments():
    from limbo.models import Attachment

    history = ToolHistory(
        [Message(role="tool", content="old", tool_call_id="c1")]
    )
    attachment = Attachment(kind="image", name="p.png", path="/tmp/p.png")

    history.record_result("c1", "new", [attachment])

    assert history.messages[0].content == "new"
    assert history.messages[0].attachments == [attachment]


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


def test_recover_without_intent_info_keeps_legacy_placeholder():
    messages = [Message(role="user", content="go"), assistant_with_calls("c1")]

    recovery = recover(messages, started=None)

    assert recovery.messages[-1].content == INTERRUPTED_CONTENT
    assert recovery.rerun == []


def test_recover_reruns_nothing_that_already_has_a_result():
    messages = [
        assistant_with_calls("c1", "c2"),
        Message(role="tool", content="done", tool_call_id="c1"),
    ]

    recovery = recover(messages, started=set(), is_safe=lambda _name: True)

    assert [tc["id"] for tc in recovery.rerun] == ["c2"]
    assert [m.tool_call_id for m in recovery.messages if m.role == "tool"] == [
        "c1",
        "c2",
    ]
