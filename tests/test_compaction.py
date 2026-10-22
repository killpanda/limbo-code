"""Tests for the pure compaction logic (no agent/UI involved)."""

from __future__ import annotations

from limbo.compaction import (
    find_split_point,
)
from limbo.models import Message


def _user(text: str) -> Message:
    return Message(role="user", content=text)


def _assistant(text: str, tool_calls: list[dict] | None = None) -> Message:
    return Message(role="assistant", content=text, tool_calls=tool_calls)


def _tool(call_id: str, text: str) -> Message:
    return Message(role="tool", tool_call_id=call_id, content=text)


def _tool_call(call_id: str, name: str = "read") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": {"path": "x"}},
    }


# -- should_compact -----------------------------------------------------------


# -- find_split_point ---------------------------------------------------------


def test_find_split_point_never_splits_tool_group():
    # Boundary would naturally land on the tool result; it must move forward
    # past the whole assistant(tool_calls) + tool group.
    system = Message(role="system", content="sys")
    history = [
        system,
        _user("start"),
        _assistant(None, tool_calls=[_tool_call("c1"), _tool_call("c2")]),
        _tool("c1", "output one"),
        _tool("c2", "output two"),
        _user("recent message"),
    ]
    # Budget that fits only the trailing user message + the tool group, with
    # the natural walking boundary landing inside the tool results.
    split = find_split_point(history, keep_recent_tokens=20)
    assert split is not None
    assert history[split].role != "tool"
    kept = history[split:]
    # If the kept region contains a tool result, its assistant call must be
    # kept too.
    tool_ids = {m.tool_call_id for m in kept if m.role == "tool"}
    call_ids = {
        tc["id"] for m in kept if m.tool_calls for tc in m.tool_calls
    }
    assert tool_ids <= call_ids


# -- build_summary_prompt -----------------------------------------------------


# -- summary message ----------------------------------------------------------


