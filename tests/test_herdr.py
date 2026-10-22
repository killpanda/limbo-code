"""Tests for the Herdr integration reporter (limbo.integrations.herdr)."""

from __future__ import annotations

import subprocess

from limbo.integrations.herdr import AGENT, SOURCE, HerdrReporter


def make_reporter(monkeypatch) -> tuple[HerdrReporter, list[list[str]]]:
    """A reporter with Popen captured; returns (reporter, recorded argv)."""
    calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))

        class _Proc:
            def wait(self, timeout=None):
                return 0

        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # Deterministic seq for assertions (production starts from the wall
    # clock; see HerdrReporter.__init__).
    return HerdrReporter("w1:p2", "/usr/local/bin/herdr", start_seq=0), calls


def test_report_state_command_and_seq(monkeypatch):
    reporter, calls = make_reporter(monkeypatch)
    reporter.report("working")
    reporter.report("idle")

    assert calls[0] == [
        "/usr/local/bin/herdr",
        "pane",
        "report-agent",
        "w1:p2",
        "--source",
        SOURCE,
        "--agent",
        AGENT,
        "--state",
        "working",
        "--seq",
        "1",
    ]
    # Strictly increasing seq across reports from the same source.
    assert calls[1][calls[1].index("--seq") + 1] == "2"
    assert calls[1][calls[1].index("--state") + 1] == "idle"


