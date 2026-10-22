"""Tests for the /tracing-viewer command logic (limbo.tracing.command)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from limbo.tracing.command import TracingViewerCommand, can_open_browser


@dataclass
class FakeAgent:
    session_id: str
    session_path: Path
    workdir: Path


@pytest.fixture
def agent(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    for stem in ("20260101-a", "20260102-b", "20260102-c"):
        (session_dir / f"{stem}.jsonl").write_text("")
    return FakeAgent("20260101-a", session_dir / "20260101-a.jsonl", tmp_path)


def make_command(agent, opened=True, available=True):
    calls: list[str] = []

    def open_browser(url: str) -> bool:
        calls.append(url)
        return opened

    command = TracingViewerCommand(lambda: agent, open_browser, lambda: available)
    return command, calls


def test_opens_current_session_in_browser(agent):
    command, calls = make_command(agent)
    try:
        message = command.run("")
        assert command.viewer is not None and command.viewer.running
        assert calls == [command.viewer.url("20260101-a")]
        assert "已在浏览器打开" in message
        # Second call reuses the same server.
        port = command.viewer.port
        command.run("")
        assert command.viewer.port == port
    finally:
        command.close()


def test_prints_url_when_no_browser(agent):
    command, calls = make_command(agent, available=False)
    try:
        message = command.run("")
        assert calls == []
        assert command.viewer is not None
        assert command.viewer.url("20260101-a") in message
        assert "ssh -L" in message
    finally:
        command.close()


def test_session_prefix_and_errors(agent):
    command, calls = make_command(agent)
    try:
        command.run("20260102-c")
        assert calls[-1].endswith("#20260102-c")
        assert "找不到会话" in command.run("nope")
        assert "不唯一" in command.run("20260102")
    finally:
        command.close()


def test_off_stops_server(agent):
    command, _ = make_command(agent)
    assert "未在运行" in command.run("off")
    command.run("")
    assert "已关闭" in command.run("off")
    assert command.viewer is None


def test_can_open_browser_respects_ssh_and_display(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 1 5.6.7.8 22")
    assert can_open_browser() is False
    monkeypatch.delenv("SSH_CONNECTION")
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert can_open_browser() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert can_open_browser() is True
