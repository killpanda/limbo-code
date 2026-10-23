"""TUI wiring of daily mode: `limbo daily`, /daily-mode, /memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.status_bar import StatusBar


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def make_app(tmp_path: Path, daily: bool = False) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
        daily=daily,
    )


def _chat_text(screen) -> str:
    chat = screen.query_one("#chat", ChatWidget)
    return "\n".join(str(w.render()) for w in chat.query("*"))


@pytest.mark.asyncio
async def test_daily_app_starts_in_daily_mode_and_toggles(tmp_path):
    app = make_app(tmp_path, daily=True)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert screen.agent.daily_mode
        assert "☀daily" in screen.query_one("#statusbar", StatusBar)._context_text()

        screen._handle_command("/daily-mode off")
        await pilot.pause()
        assert not screen.agent.daily_mode
        assert "☀daily" not in screen.query_one("#statusbar", StatusBar)._context_text()

        screen._handle_command("/memory")
        await pilot.pause()
        assert "USER.md" in _chat_text(screen)
