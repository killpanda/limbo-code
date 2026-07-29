"""Tests for the pure compaction logic (no agent/UI involved)."""

from __future__ import annotations

from limbo.compaction import (
    SUMMARY_SECTIONS,
    CompactionConfig,
    build_summary_prompt,
    estimate_tokens,
    find_split_point,
    is_summary_message,
    make_summary_message,
    should_compact,
)
from limbo.models import Message

WINDOW = 128_000


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


def test_should_compact_above_threshold():
    cfg = CompactionConfig()
    assert should_compact(WINDOW - cfg.reserve_tokens + 1, WINDOW, cfg)


def test_should_compact_at_threshold_is_false():
    cfg = CompactionConfig()
    assert not should_compact(WINDOW - cfg.reserve_tokens, WINDOW, cfg)


def test_should_compact_below_threshold():
    cfg = CompactionConfig()
    assert not should_compact(1000, WINDOW, cfg)


def test_should_compact_disabled():
    cfg = CompactionConfig(enabled=False)
    assert not should_compact(WINDOW, WINDOW, cfg)


# -- estimate_tokens ----------------------------------------------------------


def test_estimate_tokens_empty():
    assert estimate_tokens([]) == 0


def test_estimate_tokens_counts_content_and_overhead():
    # 400 chars -> 100 tokens + 4 overhead.
    assert estimate_tokens([_user("x" * 400)]) == 104


def test_estimate_tokens_includes_tool_calls_json():
    msg = _assistant(None, tool_calls=[_tool_call("c1")])
    plain = _assistant(None)
    assert estimate_tokens([msg]) > estimate_tokens([plain])


# -- find_split_point ---------------------------------------------------------


def test_find_split_point_keeps_within_budget():
    system = Message(role="system", content="sys")
    history = [system] + [_user("x" * 400) for _ in range(20)]  # ~104 tokens each
    split = find_split_point(history, keep_recent_tokens=500)
    assert split is not None
    assert estimate_tokens(history[split:]) <= 500
    # Boundary never touches the system message region.
    assert split > 1


def test_find_split_point_returns_none_when_history_too_short():
    history = [Message(role="system", content="sys"), _user("hi")]
    assert find_split_point(history, keep_recent_tokens=20_000) is None


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


def test_find_split_point_adjustment_running_off_end_returns_none():
    system = Message(role="system", content="sys")
    history = [
        system,
        _user("old stuff " * 100),
        _assistant(None, tool_calls=[_tool_call("c1")]),
        _tool("c1", "huge output " * 100),
    ]
    # Tiny budget: boundary walks onto the tool message, adjustment pushes it
    # past the end -> nothing safe to cut.
    assert find_split_point(history, keep_recent_tokens=10) is None


# -- build_summary_prompt -----------------------------------------------------


def test_build_summary_prompt_without_previous():
    prompt = build_summary_prompt([_user("do the thing"), _assistant("done")])
    assert len(prompt) == 1
    assert prompt[0].role == "user"
    text = prompt[0].content or ""
    for section in SUMMARY_SECTIONS:
        assert f"## {section}" in text
    assert "2000" in text  # soft length cap
    assert "do the thing" in text
    assert "<previous-summary>" not in text


def test_build_summary_prompt_with_previous_is_iterative():
    prompt = build_summary_prompt([_user("new work")], previous_summary="OLD SUMMARY")
    text = prompt[0].content or ""
    assert "<previous-summary>" in text
    assert "OLD SUMMARY" in text
    assert "new work" in text


def test_build_summary_prompt_truncates_huge_tool_output():
    prompt = build_summary_prompt([_tool("c1", "y" * 50_000)])
    text = prompt[0].content or ""
    assert "[... truncated ...]" in text
    assert len(text) < 50_000


# -- summary message ----------------------------------------------------------


def test_make_summary_message_roundtrip():
    msg = make_summary_message("  the summary  ")
    assert msg.role == "user"
    assert is_summary_message(msg)
    assert "the summary" in (msg.content or "")


def test_is_summary_message_rejects_normal_user_message():
    assert not is_summary_message(_user("prefix <conversation-summary>"))
    assert not is_summary_message(_user("hello"))
