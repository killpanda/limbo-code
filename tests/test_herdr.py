"""Tests for the Herdr integration reporter (limbo.integrations.herdr)."""

from __future__ import annotations

import shutil
import subprocess

from limbo.integrations.herdr import AGENT, SOURCE, HerdrReporter

FULL_ENV = {
    "HERDR_ENV": "1",
    "HERDR_PANE_ID": "w1:p2",
    "HERDR_BIN_PATH": "/usr/local/bin/herdr",
    "HERDR_SOCKET_PATH": "/tmp/herdr.sock",
}


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
    reporter = HerdrReporter.from_env(FULL_ENV)
    assert reporter is not None
    return reporter, calls


def test_from_env_outside_herdr_is_none():
    assert HerdrReporter.from_env({}) is None
    assert HerdrReporter.from_env({"HERDR_ENV": "0"}) is None


def test_from_env_requires_pane_and_bin(monkeypatch):
    # No herdr binary anywhere: even a full-looking env cannot activate.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert HerdrReporter.from_env({"HERDR_ENV": "1"}) is None
    assert (
        HerdrReporter.from_env({"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1"}) is None
    )
    assert (
        HerdrReporter.from_env(
            {"HERDR_ENV": "1", "HERDR_BIN_PATH": "/bin/herdr"}
        )
        is None
    )


def test_from_env_falls_back_to_path_lookup(monkeypatch):
    # HERDR_BIN_PATH is not guaranteed to be set; find herdr on PATH.
    monkeypatch.setattr(shutil, "which", lambda name: "/resolved/herdr")
    env = {"HERDR_ENV": "1", "HERDR_PANE_ID": "w1:p1", "HERDR_BIN_PATH": ""}
    reporter = HerdrReporter.from_env(env)
    assert isinstance(reporter, HerdrReporter)
    assert reporter._bin_path == "/resolved/herdr"


def test_from_env_inside_herdr():
    reporter = HerdrReporter.from_env(FULL_ENV)
    assert isinstance(reporter, HerdrReporter)


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


def test_report_with_message_and_session(monkeypatch):
    reporter, calls = make_reporter(monkeypatch)
    reporter.report(
        "blocked",
        message="等待确认",
        session_id="sess-1",
        session_path="/tmp/sess-1.jsonl",
    )
    cmd = calls[0]
    assert cmd[cmd.index("--message") + 1] == "等待确认"
    assert cmd[cmd.index("--agent-session-id") + 1] == "sess-1"
    assert cmd[cmd.index("--agent-session-path") + 1] == "/tmp/sess-1.jsonl"


def test_report_session_command(monkeypatch):
    reporter, calls = make_reporter(monkeypatch)
    reporter.report("idle")
    reporter.report_session("sess-2", "/tmp/sess-2.jsonl")
    cmd = calls[1]
    assert cmd[:3] == ["/usr/local/bin/herdr", "pane", "report-agent-session"]
    assert cmd[cmd.index("--agent-session-id") + 1] == "sess-2"
    assert cmd[cmd.index("--agent-session-path") + 1] == "/tmp/sess-2.jsonl"
    # Session reports share the same increasing sequence as state reports.
    assert cmd[cmd.index("--seq") + 1] == "2"


def test_release_command(monkeypatch):
    reporter, calls = make_reporter(monkeypatch)
    reporter.release()
    assert calls[0] == [
        "/usr/local/bin/herdr",
        "pane",
        "release-agent",
        "w1:p2",
        "--source",
        SOURCE,
        "--agent",
        AGENT,
    ]


def test_popen_failure_is_swallowed(monkeypatch):
    def boom(cmd, **kwargs):
        raise OSError("no such file")

    monkeypatch.setattr(subprocess, "Popen", boom)
    reporter = HerdrReporter.from_env(FULL_ENV)
    assert reporter is not None
    reporter.report("working")  # must not raise
    reporter.report_session("s")  # must not raise
    reporter.release()  # must not raise
