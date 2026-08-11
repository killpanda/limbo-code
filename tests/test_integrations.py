"""Tests for the integration abstraction layer (limbo.integrations)."""

from __future__ import annotations

import atexit
import signal

from limbo.integrations import (
    CompositeReporter,
    create_reporters,
    install_exit_hooks,
)


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


# -- install_exit_hooks ------------------------------------------------------


def test_install_exit_hooks_noop_without_reporters(monkeypatch):
    registered: list = []
    monkeypatch.setattr(atexit, "register", registered.append)
    install_exit_hooks(CompositeReporter())
    assert registered == []


def test_install_exit_hooks_registers_atexit_and_signals(monkeypatch):
    atexit_registered: list = []
    handlers: dict[int, object] = {}
    monkeypatch.setattr(atexit, "register", atexit_registered.append)
    monkeypatch.setattr(
        signal, "signal", lambda sig, handler: handlers.__setitem__(sig, handler)
    )
    composite = CompositeReporter([FakeReporter()])
    install_exit_hooks(composite)

    assert atexit_registered == [composite.release]
    assert signal.SIGTERM in handlers
    if hasattr(signal, "SIGHUP"):
        assert signal.SIGHUP in handlers


def test_signal_hook_releases_then_reraises(monkeypatch):
    reporter = FakeReporter()
    composite = CompositeReporter([reporter])
    handlers: dict[int, object] = {}
    dispositions: list[tuple[int, object]] = []
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    def fake_signal(sig, handler):
        handlers.setdefault(sig, handler)
        dispositions.append((sig, handler))

    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.setattr("os.kill", lambda pid, sig: killed.append((pid, sig)))

    install_exit_hooks(composite)
    handler = handlers[signal.SIGTERM]
    handler(signal.SIGTERM, None)

    assert reporter.calls == [("release",)]
    # Default disposition restored before re-raising the same signal.
    assert dispositions[-1] == (signal.SIGTERM, signal.SIG_DFL)
    import os

    assert killed == [(os.getpid(), signal.SIGTERM)]
