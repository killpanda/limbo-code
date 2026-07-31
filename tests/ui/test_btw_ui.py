"""Tests for the /btw overlay (limbo.ui.screens.btw + main-screen wiring)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.screens.btw import BtwScreen
from limbo.ui.screens.main import MainScreen
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


class GatedLLMClient:
    """First call (the main turn) hangs until released; later calls answer."""

    def __init__(self):
        self.calls: list[tuple[list, list]] = []
        self.release_main = asyncio.Event()

    async def chat(self, messages, tools, on_request=None):
        self.calls.append((messages, tools))
        if len(self.calls) == 1:
            await self.release_main.wait()
            yield TextChunk(text="main done")
        else:
            yield TextChunk(text="btw answer")


def make_app(tmp_path: Path, fake_llm=None) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm or FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


async def submit(pilot, text: str) -> None:
    screen = pilot.app.screen_stack[-1]
    input_widget = screen.query_one("#input", InputWidget)
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


async def wait_for(predicate, pilot, rounds: int = 100) -> bool:
    for _ in range(rounds):
        await pilot.pause()
        if predicate():
            return True
    return False


@pytest.mark.asyncio
async def test_btw_opens_overlay_and_streams_answer(tmp_path):
    fake_llm = FakeLLMClient([[TextChunk(text="it's "), TextChunk(text="config.py")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/btw which file holds settings?")

        screen = pilot.app.screen
        assert isinstance(screen, BtwScreen)
        assert await wait_for(
            lambda: fake_llm.calls == 1
            and "config.py" in screen.query_one("#btw-answer").source,
            pilot,
        )
        assert "btw · which file holds settings?" in str(
            screen.query_one("#btw-question").content
        )

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(pilot.app.screen, MainScreen)
        # Nothing leaked into the session history (still just the system msg).
        main = pilot.app.screen
        assert len(main.agent.messages) == 1


@pytest.mark.asyncio
async def test_btw_without_question_shows_usage(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/btw")
        assert isinstance(pilot.app.screen, MainScreen)
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        assert "用法：/btw" in chat.transcript_text()


@pytest.mark.asyncio
async def test_btw_works_mid_turn_without_touching_history(tmp_path):
    fake_llm = GatedLLMClient()
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "do work")
        assert await wait_for(lambda: len(fake_llm.calls) == 1, pilot)

        # Mid-turn: /btw must bypass the steer queue and open the overlay.
        await submit(pilot, "/btw quick question?")
        screen = pilot.app.screen
        assert isinstance(screen, BtwScreen)
        assert await wait_for(lambda: len(fake_llm.calls) == 2, pilot)
        # The side query ran with no tools on a history snapshot.
        _, btw_tools = fake_llm.calls[1]
        assert btw_tools == []

        await pilot.press("escape")
        await pilot.pause()
        main = pilot.app.screen
        assert isinstance(main, MainScreen)
        # History untouched by the side question (system + user so far).
        assert len(main.agent.messages) == 2

        # The main turn still completes normally afterwards.
        fake_llm.release_main.set()
        chat = main.query_one("#chat", ChatWidget)
        assert await wait_for(lambda: "main done" in chat.transcript_text(), pilot)


@pytest.mark.asyncio
async def test_btw_error_is_shown_in_overlay(tmp_path):
    class FailingClient:
        async def chat(self, messages, tools, on_request=None):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    app = make_app(tmp_path, FailingClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/btw hi?")
        screen = pilot.app.screen
        assert isinstance(screen, BtwScreen)
        assert await wait_for(
            lambda: "boom" in screen.query_one("#btw-answer").source, pilot
        )
