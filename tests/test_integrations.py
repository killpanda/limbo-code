"""Tests for the integration abstraction layer (limbo.integrations)."""

from __future__ import annotations

from limbo.integrations import CompositeReporter, create_reporters


class FakeReporter:
    def __init__(self):
        self.calls: list[tuple] = []

    def report(self, state, *, message=None, session_id=None, session_path=None):
        self.calls.append(("report", state, message, session_id, session_path))

    def report_session(self, session_id, session_path=None):
        self.calls.append(("report_session", session_id, session_path))

    def release(self):
        self.calls.append(("release",))


def test_empty_composite_is_a_safe_noop():
    composite = CompositeReporter()
    assert not composite
    composite.report("working")  # must not raise
    composite.report_session("s", "/tmp/s.jsonl")
    composite.release()


def test_composite_fans_out_to_all_reporters():
    a, b = FakeReporter(), FakeReporter()
    composite = CompositeReporter([a, b])
    assert composite

    composite.report("blocked", message="等待确认")
    composite.report_session("sess-1", "/tmp/sess-1.jsonl")
    composite.release()

    expected = [
        ("report", "blocked", "等待确认", None, None),
        ("report_session", "sess-1", "/tmp/sess-1.jsonl"),
        ("release",),
    ]
    assert a.calls == expected
    assert b.calls == expected


def test_create_reporters_outside_any_tool_is_empty():
    assert not create_reporters({})


def test_create_reporters_detects_herdr():
    composite = create_reporters(
        {
            "HERDR_ENV": "1",
            "HERDR_PANE_ID": "w1:p1",
            "HERDR_BIN_PATH": "/bin/herdr",
        }
    )
    assert composite
