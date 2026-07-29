"""Tests for the /compact command and compaction UI rendering."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from limbo.compaction import make_summary_message
from limbo.config import Config
from limbo.models import CompletionMeta, Message, TextChunk
from limbo.sessions import CompactionRecord, SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget


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
async def test_compact_command_registered(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen_stack[-1]
        assert isinstance(screen, MainScreen)
        command = screen._commands.get("/compact")
        assert command is not None
        assert command.handler is not None
        assert "/compact" in screen._commands.help_text()
        # And it shows up in the slash-menu candidates.
        assert any(c.name == "/compact" for c in screen._slash_candidates())


@pytest.mark.asyncio
async def test_compact_rejected_while_turn_is_running(tmp_path):
    """P2-1: /compact during an in-flight turn is refused with an info hint,
    and no compaction worker touches the history."""
    fake_llm = FakeLLMClient()
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen_stack[-1]
        chat = screen.query_one("#chat", ChatWidget)

        screen._agent_busy = True  # simulate an in-flight turn
        before = list(screen.agent.messages)
        screen._handle_command("/compact")
        await pilot.pause()

        assert "请等待完成后再压缩" in chat.transcript_text()
        assert screen.agent.messages == before
        assert fake_llm.calls == []


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


@pytest.mark.asyncio
async def test_compact_blocks_input_and_duplicate_compact(tmp_path):
    """P2-1: while a compaction worker is running, the input is disabled
    and a second /compact is refused — no concurrent history rewrites."""
    release = asyncio.Event()

    class SlowSummaryClient:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools, on_request=None):
            self.calls.append((messages, tools))
            await release.wait()  # hold the summary call open
            yield TextChunk(text="SUMMARY")
            yield CompletionMeta()

    client = SlowSummaryClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen_stack[-1]
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)
        agent = screen.agent
        agent.messages.append(Message(role="user", content="x" * 90_000))
        agent.messages.append(Message(role="assistant", content="y" * 90_000))
        agent.messages.append(Message(role="user", content="latest question"))

        screen._handle_command("/compact")
        await pilot.pause()
        await pilot.pause()

        # While the summary call is in flight: busy + input disabled.
        assert screen._agent_busy
        assert input_widget.disabled

        # A duplicate /compact is refused without touching the history.
        before = list(agent.messages)
        screen._handle_command("/compact")
        await pilot.pause()
        assert "请等待完成后再压缩" in chat.transcript_text()
        assert agent.messages == before
        assert len(client.calls) == 1  # no second summary call

        release.set()
        for _ in range(4):
            await pilot.pause()
        assert not screen._agent_busy
        assert not input_widget.disabled
        assert "已压缩上下文" in chat.transcript_text()


@pytest.mark.asyncio
async def test_resumed_summary_renders_as_info_not_user_bubble(tmp_path):
    session_id = "20260729-000000-000000-aa"
    summary_msg = make_summary_message("OLD SUMMARY")
    kept = Message(role="user", content="recent question")
    path = tmp_path / "sessions" / f"{session_id}.jsonl"
    save_session(
        path,
        SessionMeta(
            id=session_id,
            workdir=str(tmp_path.resolve()),
            title="compacted session",
        ),
        [Message(role="system", content="sys"), summary_msg, kept],
        compactions=[
            CompactionRecord(
                id="cmp-1",
                first_kept_entry_id=kept.id,
                summary="OLD SUMMARY",
                trigger="auto",
            )
        ],
    )

    app = make_app(tmp_path, resume=path)
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = pilot.app.screen_stack[-1].query_one("#chat", ChatWidget)
        transcript = chat.transcript_text()
        assert "（此前对话已压缩为摘要）" in transcript
        assert "OLD SUMMARY" not in transcript
        assert "recent question" in transcript
