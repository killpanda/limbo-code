"""Tests for the agent turn flow in the main screen."""

import asyncio

import pytest

from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools, on_request=None):
        for event in self.responses.pop(0):
            yield event


@pytest.mark.asyncio
async def test_tool_executes_without_confirmation(tmp_path):
    """A write tool call applies immediately; no dialog is shown."""
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
            [TextChunk(text="done")],
        ]
    )
    app = LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm,
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("write x.txt"))
        await pilot.pause()
        await pilot.pause()

        assert (tmp_path / "x.txt").read_text() == "hi"
        # The turn completes without any modal screen being pushed.
        assert pilot.app.screen_stack[-1] is main_screen
        chat = main_screen.query_one("#chat", ChatWidget)
        assert chat.tool_cards["c1"].state == "success"
        assert "done" in chat.transcript_text()


@pytest.mark.asyncio
async def test_error_event_is_shown(tmp_path):
    class ErrorLLMClient:
        async def chat(self, messages, tools, on_request=None):
            raise RuntimeError("boom")
            if False:
                yield TextChunk(text="")

    cfg = Config()
    cfg.llm.api_key = "test"
    app = LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=ErrorLLMClient(),
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("hi"))
        await pilot.pause()

        chat = main_screen.query_one("#chat", ChatWidget)
        rendered = chat.transcript_text()
        assert "LLM error" in rendered
        assert "boom" in rendered


@pytest.mark.asyncio
async def test_tool_error_is_shown(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="read", arguments={"path": "missing.txt"})],
        ]
    )
    app = LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm,
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("read missing"))
        await pilot.pause()

        chat = main_screen.query_one("#chat", ChatWidget)
        card = chat.tool_cards["c1"]
        assert card.state == "error"


@pytest.mark.asyncio
async def test_input_disabled_during_streaming(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"

    hold = asyncio.Event()

    class HoldingLLMClient:
        async def chat(self, messages, tools, on_request=None):
            yield TextChunk(text="hello")
            await hold.wait()
            yield TextChunk(text=" world")

    app = LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=HoldingLLMClient(),
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("hi"))
        await pilot.pause()

        input_widget = main_screen.query_one("#input", InputWidget)
        assert input_widget.disabled is True

        hold.set()
        await pilot.pause()
        await pilot.pause()

        assert input_widget.disabled is False
