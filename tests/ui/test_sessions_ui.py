"""Tests for session management UI: slash commands, picker, resume rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import Message
from limbo.sessions import SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget
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


