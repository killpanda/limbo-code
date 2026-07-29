"""Snapshot tests for the main UI states (RFC LIM-16 v1.1 Test Plan §7.1).

Four scenarios: fresh idle screen, thinking + running tool card, tool error
+ structured error message, session picker modal. Run
``pytest --snapshot-update`` after intentional visual changes and review the
SVG diff.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from limbo.config import Config
from limbo.models import Message
from limbo.sessions import SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.screens.session_picker import SessionPicker
from limbo.ui.widgets.chat import ChatWidget


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover - make this an async generator


# Fixed workdir so the status bar renders identical text across runs
# (tmp_path contains a random pytest-N segment that would break snapshots).
SNAPSHOT_WORKDIR = Path("/tmp/limbo-snapshot-workdir")


@pytest.fixture(autouse=True)
def _deterministic_ui(monkeypatch):
    """Freeze widget clocks and disable the spinner timer for stable snapshots.

    Replaces the ``time`` *name inside the widget modules'* namespaces (not
    the global time module — freezing that would stall Textual's own clock).
    """
    SNAPSHOT_WORKDIR.mkdir(parents=True, exist_ok=True)
    frozen = SimpleNamespace(monotonic=lambda: 100.0)
    monkeypatch.setattr("limbo.ui.widgets.status_bar.time", frozen)
    monkeypatch.setattr("limbo.ui.widgets.tool_card.time", frozen)
    from limbo.ui.widgets.status_bar import StatusBar

    monkeypatch.setattr(StatusBar, "_tick", lambda self: None)


def make_app(tmp_path: Path) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=SNAPSHOT_WORKDIR,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


def test_snapshot_idle(snap_compare, tmp_path):
    app = make_app(tmp_path)
    assert snap_compare(app, terminal_size=(90, 28))


def test_snapshot_thinking_and_running_tool(snap_compare, tmp_path):
    app = make_app(tmp_path)

    async def run_before(pilot) -> None:
        await pilot.pause()
        screen = pilot.app.screen
        statusbar = screen.query_one("#statusbar")
        statusbar.set_state("thinking…", "thinking")
        chat = screen.query_one("#chat", ChatWidget)
        chat.add_user_message("帮我看一下 agent.py 的实现")
        chat.add_tool_card("c1", "read", {"path": "src/limbo/agent.py"})
        await pilot.pause()

    assert snap_compare(app, terminal_size=(90, 28), run_before=run_before)


def test_snapshot_tool_error_and_error_message(snap_compare, tmp_path):
    app = make_app(tmp_path)

    async def run_before(pilot) -> None:
        await pilot.pause()
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        chat.add_user_message("删掉 build 目录")
        card = chat.add_tool_card("c1", "bash", {"command": "rm -rf build/"})
        card.set_error("permission denied")
        chat.add_error("LLM 请求失败：HTTP 429 rate limited\n请稍后重试")
        await pilot.pause()

    assert snap_compare(app, terminal_size=(90, 28), run_before=run_before)


def test_snapshot_session_picker(snap_compare, tmp_path, monkeypatch):
    # save_session re-stamps updated_at with wall-clock time; freeze it so the
    # picker's "title · updated · id" row is identical across runs.
    monkeypatch.setattr(
        "limbo.sessions._utc_now_iso", lambda: "2026-07-30T10:00:00+00:00"
    )
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    path = session_dir / "s1.jsonl"
    save_session(
        path,
        SessionMeta(
            id="s1",
            workdir=str(SNAPSHOT_WORKDIR.resolve()),
            title="旧会话",
            updated_at="2026-07-30T10:00:00+00:00",
        ),
        [Message(role="system", content="sys"), Message(role="user", content="hi")],
    )
    app = make_app(tmp_path)

    async def run_before(pilot) -> None:
        await pilot.pause()
        from limbo.sessions import list_sessions

        sessions = list_sessions(session_dir, workdir=SNAPSHOT_WORKDIR)
        await pilot.app.push_screen(SessionPicker(sessions))
        await pilot.pause()

    assert snap_compare(app, terminal_size=(90, 28), run_before=run_before)
