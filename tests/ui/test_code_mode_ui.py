"""Tests for the /code-mode UI toggle (Code Mode presentation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.status_bar import StatusBar


class FakeClient:
    async def chat(self, messages, tools, on_request=None):
        if on_request is not None:
            on_request({"model": "fake", "messages": [], "tools": tools})
        if False:
            yield


def make_app(tmp_path: Path) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeClient(),
        session_dir=tmp_path / "sessions",
    )


def get_screen(pilot) -> MainScreen:
    return pilot.app.screen_stack[-1]


@pytest.mark.asyncio
async def test_code_mode_on_off_toggle(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)

        assert screen.agent.code_mode is False
        assert "⌨code" not in statusbar._context_text()

        screen._handle_command("/code-mode on")
        await pilot.pause()
        assert screen.agent.code_mode is True
        assert "⌨code" in statusbar._context_text()
        assert "Code Mode 已开启" in chat.transcript_text()
        # The wire catalog collapses.
        assert [d["function"]["name"] for d in screen.agent.registry.definitions()] == [
            "run_code"
        ]

        screen._handle_command("/code-mode off")
        await pilot.pause()
        assert screen.agent.code_mode is False
        assert "⌨code" not in statusbar._context_text()
        assert "Code Mode 已关闭" in chat.transcript_text()
        assert "run_code" not in {
            d["function"]["name"] for d in screen.agent.registry.definitions()
        }


@pytest.mark.asyncio
async def test_code_mode_no_arg_flips(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        assert screen.agent.code_mode is False
        screen._handle_command("/code-mode")
        await pilot.pause()
        assert screen.agent.code_mode is True
        screen._handle_command("/code-mode")
        await pilot.pause()
        assert screen.agent.code_mode is False


@pytest.mark.asyncio
async def test_code_mode_bad_arg_usage_hint(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        screen._handle_command("/code-mode maybe")
        await pilot.pause()
        assert screen.agent.code_mode is False
        assert "用法" in chat.transcript_text()


@pytest.mark.asyncio
async def test_code_mode_rejected_while_busy(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        screen.pump._running = True  # simulate an in-flight turn
        screen._handle_command("/code-mode on")
        await pilot.pause()
        assert screen.agent.code_mode is False
        assert "当前任务进行中" in chat.transcript_text()


@pytest.mark.asyncio
async def test_code_mode_in_help_and_menu(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        screen._handle_command("/help")
        await pilot.pause()
        chat = screen.query_one("#chat", ChatWidget)
        assert "/code-mode" in chat.transcript_text()
