"""Tests for the /compact command and compaction UI rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import CompletionMeta, Message, TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append((messages, tools))
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


@pytest.mark.asyncio
async def test_compact_runs_when_idle_and_reports(tmp_path):
    fake_llm = FakeLLMClient(
        [[TextChunk(text="SUMMARY"), CompletionMeta()]],
    )
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen_stack[-1]
        chat = screen.query_one("#chat", ChatWidget)
        agent = screen.agent
        # Seed enough history for a manual compaction (keep_recent=20000
        # default, so make it chunky).
        agent.messages.append(Message(role="user", content="x" * 90_000))
        agent.messages.append(Message(role="assistant", content="y" * 90_000))
        agent.messages.append(Message(role="user", content="latest question"))

        screen._handle_command("/compact")
        for _ in range(4):
            await pilot.pause()

        assert "已压缩上下文" in chat.transcript_text()
        assert fake_llm.calls and fake_llm.calls[0][1] == []


