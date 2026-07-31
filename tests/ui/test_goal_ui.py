"""UI tests for /goal: command mounting, busy semantics, badge (LIM-40)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import limbo.goal_driver as goal_driver_module
from limbo.config import Config
from limbo.goal import VerifyResult
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget
from limbo.ui.widgets.status_bar import StatusBar


class GatedClient:
    """LLM client replaying scripted events; each call may wait on a gate."""

    def __init__(self):
        self.calls = []
        self.script = []

    def add(self, events, gate=None):
        self.script.append((gate, events))

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))
        gate, events = self.script.pop(0)
        if gate is not None:
            await gate.wait()
        for event in events:
            yield event


def make_app(tmp_path: Path, client) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=client,
        session_dir=tmp_path / "sessions",
    )


def get_screen(pilot) -> MainScreen:
    return pilot.app.screen_stack[-1]


async def wait_until(pilot, predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError("timed out waiting for condition")


async def type_and_submit(pilot, input_widget: InputWidget, text: str) -> None:
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_goal_query_without_goal(tmp_path):
    client = GatedClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await type_and_submit(pilot, input_widget, "/goal")
        assert "当前没有 goal" in chat.transcript_text()
        assert len(client.calls) == 0  # query never starts a turn


@pytest.mark.asyncio
async def test_goal_set_starts_turn_and_shows_badge(tmp_path, monkeypatch):
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="working on it")], gate=gate)
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)

        async def passing(command, workdir, **kwargs):
            return VerifyResult(exit_code=0)

        monkeypatch.setattr(goal_driver_module, "run_verify", passing)

        await type_and_submit(pilot, input_widget, "/goal 补齐测试")
        await wait_until(pilot, lambda: len(client.calls) == 1)
        # Set the verify command while the first turn is still running.
        await type_and_submit(pilot, input_widget, "/goal verify pytest -x")
        gate.set()
        await wait_until(pilot, lambda: "目标达成" in chat.transcript_text())

        transcript = chat.transcript_text()
        assert "❯ /goal 补齐测试" in transcript
        assert "验收命令已设置" in transcript
        assert "第 1/10 轮验收" in transcript
        # Goal passed → badge hidden, state persisted.
        assert statusbar._goal is None
        state = screen.driver.status()
        assert state is not None and state.status == "passed"
        # The first turn's prompt is the goal block, not the raw command.
        assert "🎯 Goal 模式已启动" in client.calls[0][-1].content


@pytest.mark.asyncio
async def test_goal_badge_rainbow_while_running(tmp_path, monkeypatch):
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="still working")], gate=gate)
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        statusbar = screen.query_one("#statusbar", StatusBar)

        screen.driver.set_goal("g", verify_command="pytest")
        screen.run_worker(screen._handle_turn("goal prompt"))
        await wait_until(pilot, lambda: len(client.calls) == 1)

        # While the goal loop is executing, the badge shows rainbow mode.
        assert statusbar._goal == (0, 10)
        assert statusbar._goal_running

        gate.set()
        await wait_until(pilot, lambda: not screen.driver.running)
        # No auto-verify happened mid-test? It did (turn ended) — but the
        # point here is the badge leaves rainbow mode once the loop stops.
        assert not statusbar._goal_running


@pytest.mark.asyncio
async def test_goal_set_rejected_while_busy(tmp_path):
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="slow")], gate=gate)
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("do something"))
        await wait_until(pilot, lambda: len(client.calls) == 1)

        await type_and_submit(pilot, input_widget, "/goal 新目标")
        assert "请等待完成后再设定 /goal" in chat.transcript_text()
        assert screen.driver.status() is None

        # Query and clear still work mid-turn (allow_when_busy).
        await type_and_submit(pilot, input_widget, "/goal")
        assert "当前没有 goal" in chat.transcript_text()
        gate.set()
        await wait_until(pilot, lambda: not screen._agent_busy)


@pytest.mark.asyncio
async def test_goal_verify_rejects_dangerous_command(tmp_path):
    client = GatedClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.driver.set_goal("g")
        await type_and_submit(pilot, input_widget, "/goal verify rm -rf build/")
        assert "危险命令" in chat.transcript_text()
        state = screen.driver.status()
        assert state is not None and state.verify_command == ""
