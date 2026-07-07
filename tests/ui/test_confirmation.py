"""Tests for the confirmation flow in the main screen."""

import asyncio

import pytest

from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.confirm import ConfirmDialog
from limbo.ui.widgets.file_preview import FilePreviewWidget
from limbo.ui.widgets.input import InputWidget
from limbo.ui.widgets.sidebar import SidebarWidget


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools):
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

        main_screen.run_worker(main_screen._handle_turn("write y.txt"))
        await pilot.pause()

        await pilot.click("#reject")
        await pilot.pause()

        assert not (tmp_path / "y.txt").exists()


@pytest.mark.asyncio
async def test_error_event_is_shown(tmp_path):
    class ErrorLLMClient:
        async def chat(self, messages, tools):
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

        # Re-query with the concrete widget class for type checking.
        from limbo.ui.widgets.chat import ChatWidget

        chat = main_screen.query_one("#chat", ChatWidget)
        rendered = " ".join(str(m.content) for m in chat.messages)
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
        rendered = " ".join(str(m.content) for m in chat.messages)
        assert "read failed" in rendered

        preview = main_screen.query_one("#preview", FilePreviewWidget)
        assert "missing.txt" in preview.content


@pytest.mark.asyncio
async def test_confirm_dialog_disables_input(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"

    hold_second_response = asyncio.Event()

    class HoldingFakeLLMClient:
        def __init__(self):
            self.first = True

        async def chat(self, messages, tools):
            if self.first:
                self.first = False
                yield ToolCallEvent(
                    id="c1", name="write", arguments={"path": "x.txt", "content": "hi"}
                )
            else:
                yield TextChunk(text="stream")
                await hold_second_response.wait()
                yield TextChunk(text=" done")

    app = LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=HoldingFakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen_stack[-1]
        assert isinstance(main_screen, MainScreen)

        main_screen.run_worker(main_screen._handle_turn("write x.txt"))
        await pilot.pause()

        input_widget = main_screen.query_one("#input", InputWidget)
        assert input_widget.disabled is True
        assert any(isinstance(s, ConfirmDialog) for s in pilot.app.screen_stack)

        await pilot.click("#apply")
        await pilot.pause()

        # Input must stay disabled while the agent streams the follow-up response.
        assert input_widget.disabled is True

        hold_second_response.set()
        await pilot.pause()
        await pilot.pause()

        assert input_widget.disabled is False


@pytest.mark.asyncio
async def test_confirm_timeout_dismisses_dialog(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.ui.screens.main.CONFIRMATION_TIMEOUT", 0.2)
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
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

        assert any(isinstance(s, ConfirmDialog) for s in pilot.app.screen_stack)

        # Wait for the short timeout to elapse.
        await asyncio.sleep(0.3)
        await pilot.pause()

        assert not any(isinstance(s, ConfirmDialog) for s in pilot.app.screen_stack)
        assert not (tmp_path / "x.txt").exists()

        input_widget = main_screen.query_one("#input", InputWidget)
        assert input_widget.disabled is False


@pytest.mark.asyncio
async def test_confirm_dialog_resumes_conversation_loop(tmp_path):
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

        await pilot.click("#apply")
        await pilot.pause()
        await pilot.pause()

        assert (tmp_path / "x.txt").read_text() == "hi"
        chat = main_screen.query_one("#chat", ChatWidget)
        rendered = " ".join(str(m.content) for m in chat.messages)
        assert "done" in rendered


@pytest.mark.asyncio
async def test_reject_clears_pending_tool(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
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

        await pilot.click("#reject")
        await pilot.pause()

        assert not (tmp_path / "x.txt").exists()
        assert main_screen.agent._pending_tool is None


@pytest.mark.asyncio
async def test_timeout_clears_pending_tool(tmp_path, monkeypatch):
    monkeypatch.setattr("limbo.ui.screens.main.CONFIRMATION_TIMEOUT", 0.2)
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
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

        await asyncio.sleep(0.3)
        await pilot.pause()

        assert not (tmp_path / "x.txt").exists()
        assert main_screen.agent._pending_tool is None


@pytest.mark.asyncio
async def test_input_disabled_during_streaming(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"

    hold = asyncio.Event()

    class HoldingLLMClient:
        async def chat(self, messages, tools):
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


@pytest.mark.asyncio
async def test_recent_file_path_is_normalized(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "x.txt").write_text("hello")
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="read", arguments={"path": "./subdir/../x.txt"})],
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

        main_screen.run_worker(main_screen._handle_turn("read x.txt"))
        await pilot.pause()

        sidebar = main_screen.query_one("#sidebar", SidebarWidget)
        assert "x.txt" in sidebar._recent_files
        assert "subdir/../x.txt" not in sidebar._recent_files


@pytest.mark.asyncio
async def test_multi_confirmation_resumes_without_extra_user_message(tmp_path):
    """A turn with two confirmation-required tools should continue after both."""
    cfg = Config()
    cfg.llm.api_key = "test"
    fake_llm = FakeLLMClient(
        [
            [
                ToolCallEvent(id="c1", name="write", arguments={"path": "a.txt", "content": "a"}),
                ToolCallEvent(id="c2", name="write", arguments={"path": "b.txt", "content": "b"}),
            ],
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

        main_screen.run_worker(main_screen._handle_turn("write a and b"))
        await pilot.pause()

        # First confirmation dialog should be visible.
        assert any(isinstance(s, ConfirmDialog) for s in pilot.app.screen_stack)
        await pilot.click("#apply")
        await pilot.pause()
        await pilot.pause()

        # Second confirmation dialog should appear automatically.
        assert any(isinstance(s, ConfirmDialog) for s in pilot.app.screen_stack)
        await pilot.click("#apply")
        await pilot.pause()
        await pilot.pause()

        # Both files should be written and the follow-up response shown.
        assert (tmp_path / "a.txt").read_text() == "a"
        assert (tmp_path / "b.txt").read_text() == "b"
        chat = main_screen.query_one("#chat", ChatWidget)
        rendered = " ".join(str(m.content) for m in chat.messages)
        assert "done" in rendered
