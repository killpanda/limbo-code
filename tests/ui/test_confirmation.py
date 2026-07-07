"""Tests for the confirmation flow in the main screen."""

import pytest

from limbo.config import Config
from limbo.models import ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    def chat(self, messages, tools):
        for event in self.responses.pop(0):
            yield event


@pytest.mark.asyncio
async def test_confirm_dialog_applies_tool(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
        ]
    )
    app = LimboApp(workdir=tmp_path, config=cfg, llm_client=fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("write x.txt"))
        await pilot.pause()

        await pilot.click("#apply")
        await pilot.pause()

        assert (tmp_path / "x.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_confirm_dialog_rejects_tool(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="write", arguments={"path": "y.txt", "content": "hi"})],
        ]
    )
    app = LimboApp(workdir=tmp_path, config=cfg, llm_client=fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("write y.txt"))
        await pilot.pause()

        await pilot.click("#reject")
        await pilot.pause()

        assert not (tmp_path / "y.txt").exists()
