"""Sidebar + responsive layout (Limbo Dusk design, step 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.sidebar import (
    WIDE_MIN_WIDTH,
    Sidebar,
)


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def make_app(tmp_path: Path) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


@pytest.mark.asyncio
async def test_wide_terminal_shows_sidebar(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(WIDE_MIN_WIDTH + 20, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert not screen.has_class("-compact")
        assert screen.query_one("#sidebar", Sidebar).display


@pytest.mark.asyncio
async def test_command_palette_filters_tabs_and_runs(tmp_path):
    from limbo.ui.screens.command_palette import CommandPalette

    app = make_app(tmp_path)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        main = pilot.app.screen
        await pilot.press("ctrl+p")
        await pilot.pause()
        palette = pilot.app.screen
        assert isinstance(palette, CommandPalette)
        assert palette.tab == "system"
        await pilot.press(*"mode")
        await pilot.pause()
        assert palette.shown[0].value in ("/model", "/code-mode")
        await pilot.press("tab")
        await pilot.pause()
        assert palette.tab == "sessions"
        await pilot.press("shift+tab")
        await pilot.press("ctrl+a")
        for _ in range(4):
            await pilot.press("backspace")
        await pilot.press(*"/model")
        await pilot.pause()
        assert palette.shown[0].value == "/model"
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.screen is main
        # /model takes arguments, so it is prefilled rather than run.
        assert main.query_one("#input").text == "/model "


