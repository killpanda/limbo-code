"""Tests for the slash-command autocomplete menu."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.command_menu import SlashCommandMenu
from limbo.ui.widgets.input import InputWidget


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    async def chat(self, messages, tools, on_request=None):
        self.calls += 1
        if self.responses:
            for event in self.responses.pop(0):
                yield event


def make_app(tmp_path: Path, fake_llm=None) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm or FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


def get_menu(app) -> SlashCommandMenu:
    return app.screen.query_one("#slash-menu", SlashCommandMenu)


def get_input(app) -> InputWidget:
    return app.screen.query_one("#input", InputWidget)


@pytest.mark.asyncio
async def test_enter_completes_arg_command_without_executing(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "e", "x")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert not get_menu(app).is_open
        assert get_input(app).text == "/export "
        # Nothing executed yet: no export info line in the chat.
        chat = app.screen.query_one("#chat", ChatWidget)
        assert "已导出" not in chat.transcript_text()


