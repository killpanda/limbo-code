"""Tests for OSC 7501 program-status reporting (limbo.integrations.osc7501)."""

from __future__ import annotations

import base64

from limbo.agent import ErrorEvent, InterruptEvent
from limbo.integrations import TurnOutcome, create_reporters
from limbo.integrations.osc7501 import OscStatusReporter


def test_reports_encode_spec_sequences():
    out: list[str] = []
    reporter = OscStatusReporter(out.append)
    reporter.report("working")
    reporter.report("blocked", message="等待确认\n/goal")
    reporter.release()
    reporter.release()  # idempotent across exit hooks

    msg = base64.b64encode("等待确认 /goal".encode()).decode()
    assert out == [
        "\x1b]7501;state=working:app=limbo\x1b\\",
        f"\x1b]7501;state=blocked:app=limbo:kind=question:msg={msg}\x1b\\",
        "\x1b]7501;state=clear\x1b\\",
    ]


def test_guarded_by_terminal_and_env():
    out: list[str] = []
    assert not create_reporters({"TERM": "xterm"})  # no TTY writer
    assert not create_reporters({"TERM": "dumb"}, terminal=out.append)
    assert not create_reporters(
        {"LIMBO_PROGRAM_STATUS": "0"}, terminal=out.append
    )
    tmux = create_reporters({"TMUX": "/tmp/tmux-1/default"}, terminal=out.append)
    tmux.report("idle")
    assert out == ["\x1bPtmux;\x1b\x1b]7501;state=idle:app=limbo\x1b\x1b\\\x1b\\"]


def test_turn_outcome():
    outcome = TurnOutcome()
    assert outcome.state == "done"
    outcome.observe(ErrorEvent(message="boom"))
    assert (outcome.state, outcome.message) == ("error", "boom")
    interrupted = TurnOutcome()
    interrupted.observe(InterruptEvent(phase="llm_stream"))
    interrupted.observe(ErrorEvent(message="cancelled"))
    assert interrupted.state == "idle"
