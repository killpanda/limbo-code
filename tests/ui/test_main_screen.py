"""Tests for the agent turn flow in the main screen."""


import pytest

from limbo.config import Config
from limbo.models import ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools, on_request=None):
        for event in self.responses.pop(0):
            yield event


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


