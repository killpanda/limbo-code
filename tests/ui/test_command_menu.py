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

    async def chat(self, messages, tools):
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
async def test_typing_slash_opens_menu_with_all_commands(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not get_menu(app).is_open

        await pilot.press("/")
        await pilot.pause()

        menu = get_menu(app)
        assert menu.is_open
        prompts = [str(o.prompt) for o in menu.options]
        assert any("/sessions" in p for p in prompts)
        assert any("/new" in p for p in prompts)
        assert any("/export" in p for p in prompts)
        assert any("/help" in p for p in prompts)


@pytest.mark.asyncio
async def test_typing_prefix_filters_commands(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "s", "e")
        await pilot.pause()

        menu = get_menu(app)
        assert menu.is_open
        prompts = [str(o.prompt) for o in menu.options]
        assert len(prompts) == 1
        assert "/sessions" in prompts[0]


@pytest.mark.asyncio
async def test_menu_hides_when_slash_removed(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        assert get_menu(app).is_open

        await pilot.press("backspace")
        await pilot.pause()
        assert not get_menu(app).is_open


@pytest.mark.asyncio
async def test_menu_hides_after_space(tmp_path):
    """Once the user types arguments, the menu closes."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "e", "x", "p", "o", "r", "t", "space")
        await pilot.pause()

        assert not get_menu(app).is_open


@pytest.mark.asyncio
async def test_enter_executes_highlighted_no_arg_command(tmp_path):
    fake_llm = FakeLLMClient()
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        # "/he" narrows to /help; Enter executes it directly.
        await pilot.press("/", "h", "e")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert not get_menu(app).is_open
        assert get_input(app).text == ""
        chat = app.screen.query_one("#chat", ChatWidget)
        assert "/sessions" in chat.transcript_text()
        assert fake_llm.calls == 0  # help is local, nothing sent to the LLM


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


@pytest.mark.asyncio
async def test_tab_completes_highlighted_command(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "n")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        assert get_input(app).text == "/new "
        assert not get_menu(app).is_open


@pytest.mark.asyncio
async def test_down_then_enter_executes_second_command(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        # First item is /sessions; move down to /new and execute it.
        old_id = app.screen.agent.session_id
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        chat = app.screen.query_one("#chat", ChatWidget)
        assert "新会话" in chat.transcript_text()
        assert app.screen.agent.session_id != old_id


@pytest.mark.asyncio
async def test_escape_closes_menu_and_keeps_text(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("/", "s")
        await pilot.pause()
        assert get_menu(app).is_open

        await pilot.press("escape")
        await pilot.pause()

        assert not get_menu(app).is_open
        assert get_input(app).text == "/s"


@pytest.mark.asyncio
async def test_plain_message_still_reaches_llm(tmp_path):
    """Regression: normal (non-slash) input is unaffected by the menu."""
    from limbo.models import TextChunk

    fake_llm = FakeLLMClient([[TextChunk(text="pong")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        input_widget = get_input(app)
        input_widget.text = "ping"
        await pilot.press("enter")
        chat = app.screen.query_one("#chat", ChatWidget)
        for _ in range(200):
            await pilot.pause()
            if "pong" in chat.transcript_text() and not input_widget.disabled:
                break

        assert fake_llm.calls == 1
        assert not get_menu(app).is_open
