"""Tests for session management UI: slash commands, picker, resume rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import Message, TextChunk
from limbo.sessions import SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.screens.session_picker import SessionPicker
from limbo.ui.widgets.chat import ChatWidget
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


def make_app(tmp_path: Path, fake_llm=None, **kwargs) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm or FakeLLMClient(),
        session_dir=tmp_path / "sessions",
        **kwargs,
    )


def make_session(tmp_path: Path, session_id: str, user_text: str) -> Path:
    path = tmp_path / "sessions" / f"{session_id}.jsonl"
    save_session(
        path,
        SessionMeta(
            id=session_id,
            workdir=str(tmp_path.resolve()),
            title=user_text,
        ),
        [
            Message(role="system", content="sys"),
            Message(role="user", content=user_text),
            Message(role="assistant", content="old answer"),
            Message(role="tool", content="raw tool output", tool_call_id="c1"),
        ],
    )
    return path


async def submit(pilot, text: str) -> None:
    screen = pilot.app.screen_stack[-1]
    input_widget = screen.query_one("#input", InputWidget)
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


async def submit_and_wait(pilot, text: str, needle: str) -> None:
    """Submit a message and wait until `needle` appears in the chat flow."""
    chat = pilot.app.screen.query_one("#chat", ChatWidget)
    await submit(pilot, text)
    for _ in range(200):
        await pilot.pause()
        if needle in chat.transcript_text():
            return
    raise AssertionError(f"timed out waiting for {needle!r} in chat")


@pytest.mark.asyncio
async def test_help_command_does_not_call_llm(tmp_path):
    fake_llm = FakeLLMClient()
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/help")

        assert fake_llm.calls == 0
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        assert "/sessions" in chat.transcript_text()


@pytest.mark.asyncio
async def test_unknown_command_shows_hint(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/bogus")

        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        assert "/bogus" in chat.transcript_text()
        assert "/help" in chat.transcript_text()


@pytest.mark.asyncio
async def test_sessions_command_opens_picker(tmp_path):
    make_session(tmp_path, "abc123", "old conversation")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/sessions")

        assert isinstance(pilot.app.screen, SessionPicker)


@pytest.mark.asyncio
async def test_sessions_command_with_no_sessions_shows_info(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/sessions")

        assert isinstance(pilot.app.screen, MainScreen)
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        assert "没有" in chat.transcript_text()


@pytest.mark.asyncio
async def test_picker_select_switches_session(tmp_path):
    make_session(tmp_path, "abc123", "old conversation")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen
        old_id = main_screen.agent.session_id

        await submit(pilot, "/sessions")
        await pilot.press("enter")  # select the only session
        await pilot.pause()

        assert isinstance(pilot.app.screen, MainScreen)
        agent = main_screen.agent
        assert agent.session_id == "abc123" != old_id
        chat = main_screen.query_one("#chat", ChatWidget)
        transcript = chat.transcript_text()
        assert "old conversation" in transcript
        assert "old answer" in transcript
        # Raw tool outputs are not re-rendered as cards.
        assert "raw tool output" not in transcript


@pytest.mark.asyncio
async def test_picker_escape_keeps_current_session(tmp_path):
    make_session(tmp_path, "abc123", "old conversation")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen
        old_id = main_screen.agent.session_id

        await submit(pilot, "/sessions")
        await pilot.press("escape")
        await pilot.pause()

        assert isinstance(pilot.app.screen, MainScreen)
        assert main_screen.agent.session_id == old_id


@pytest.mark.asyncio
async def test_new_command_starts_fresh_session(tmp_path):
    fake_llm = FakeLLMClient([[TextChunk(text="hi there")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen
        old_id = main_screen.agent.session_id
        await submit_and_wait(pilot, "hello", "hi there")

        await submit(pilot, "/new")

        agent = main_screen.agent
        assert agent.session_id != old_id
        assert [m.role for m in agent.messages] == ["system"]
        chat = main_screen.query_one("#chat", ChatWidget)
        assert "hello" not in chat.transcript_text()


@pytest.mark.asyncio
async def test_export_command_writes_markdown(tmp_path):
    fake_llm = FakeLLMClient([[TextChunk(text="the answer")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit_and_wait(pilot, "a question", "the answer")

        out = tmp_path / "out.md"
        await submit(pilot, f"/export {out}")

        text = out.read_text()
        assert "a question" in text
        assert "the answer" in text
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        assert str(out) in chat.transcript_text()


@pytest.mark.asyncio
async def test_resume_on_startup_renders_history(tmp_path):
    session_path = make_session(tmp_path, "abc123", "old conversation")
    app = make_app(tmp_path, resume=session_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        main_screen = pilot.app.screen

        assert main_screen.agent.session_id == "abc123"
        chat = main_screen.query_one("#chat", ChatWidget)
        transcript = chat.transcript_text()
        assert "old conversation" in transcript
        assert "已恢复" in transcript
