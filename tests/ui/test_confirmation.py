"""Tests for the confirmation flow in the main screen."""

import pytest

from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools):
        for event in self.responses.pop(0):
            yield event


@pytest.mark.asyncio
async def test_confirm_dialog_applies_tool(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.agent.Path.home", lambda: tmp_path)
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
async def test_confirm_dialog_rejects_tool(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.agent.Path.home", lambda: tmp_path)
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


@pytest.mark.asyncio
async def test_confirm_writes_false_skips_dialog(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.agent.Path.home", lambda: tmp_path)
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.ui.confirm_writes = False
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

        assert (tmp_path / "x.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_confirm_edits_false_skips_dialog(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.agent.Path.home", lambda: tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.ui.confirm_edits = False
    fake_llm = FakeLLMClient(
        [
            [
                ToolCallEvent(
                    id="c1",
                    name="edit",
                    arguments={
                        "path": "a.py",
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    },
                )
            ],
        ]
    )
    app = LimboApp(workdir=tmp_path, config=cfg, llm_client=fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("edit a.py"))
        await pilot.pause()

        assert (tmp_path / "a.py").read_text() == "x = 2\n"


@pytest.mark.asyncio
async def test_error_event_is_shown(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.agent.Path.home", lambda: tmp_path)

    class ErrorLLMClient:
        async def chat(self, messages, tools):
            raise RuntimeError("boom")
            if False:
                yield TextChunk(text="")

    cfg = Config()
    cfg.llm.api_key = "test"
    app = LimboApp(workdir=tmp_path, config=cfg, llm_client=ErrorLLMClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("hi"))
        await pilot.pause()

        chat = main_screen.query_one("#chat", expect_type=type(main_screen.query_one("#chat")))
        # Re-query with the concrete widget class for type checking.
        from limbo.ui.widgets.chat import ChatWidget

        chat = main_screen.query_one("#chat", ChatWidget)
        rendered = " ".join(str(m.content) for m in chat.messages)
        assert "LLM error" in rendered
        assert "boom" in rendered
