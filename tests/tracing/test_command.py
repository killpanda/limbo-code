"""Tests for the /tracing-viewer command logic (limbo.tracing.command)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from limbo.tracing.command import TracingViewerCommand


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


