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
from limbo.ui.screens.verify_picker import VerifyPicker
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


async def wait_picker(pilot, attempts: int = 200) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if pilot.app.screen_stack and isinstance(pilot.app.screen_stack[-1], VerifyPicker):
            return
    raise AssertionError("timed out waiting for VerifyPicker")


async def wait_main(pilot, attempts: int = 200) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if isinstance(pilot.app.screen_stack[-1], MainScreen):
            return
    raise AssertionError("timed out waiting for MainScreen")


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
async def test_goal_proposal_accept_starts_closed_loop(tmp_path, monkeypatch):
    """M2 flow: /goal → proposal round → picker pops → accept → closed loop."""
    client = GatedClient()
    client.add([TextChunk(
        text="仓库是 unittest 布局。\n<verify_proposal>\n"
        "<command>python3 -m unittest discover -v</command>\n"
        "<rationale>unittest 布局</rationale>\n</verify_proposal>"
    )])
    client.add([TextChunk(text="fixing")])
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
        # Proposal round ends → confirmation picker pops.
        await wait_picker(pilot)
        await pilot.press("enter")  # accept the first candidate
        await wait_until(pilot, lambda: "目标达成" in chat.transcript_text())

        transcript = chat.transcript_text()
        assert "❯ /goal 补齐测试" in transcript
        assert "验收方式已确认：python3 -m unittest discover -v" in transcript
        assert "第 1/10 轮验收" in transcript
        assert statusbar._goal is None  # passed → badge hidden
        state = screen.driver.status()
        assert state is not None and state.status == "passed"
        assert state.verify_command == "python3 -m unittest discover -v"
        # First turn is the proposal round (proposer prompt, not work prompt).
        assert "先不要动手" in client.calls[0][-1].content


@pytest.mark.asyncio
async def test_goal_proposal_skip_keeps_single_turn(tmp_path, monkeypatch):
    client = GatedClient()
    client.add([TextChunk(
        text="<verify_proposal><command>pytest -x</command></verify_proposal>"
    )])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await type_and_submit(pilot, input_widget, "/goal 补齐测试")
        await wait_picker(pilot)
        await pilot.press("escape")  # skip
        await wait_until(
            pilot, lambda: "已跳过自动验收" in chat.transcript_text()
        )
        state = screen.driver.status()
        assert state is not None and state.verify_command == ""
        assert len(client.calls) == 1  # no work turn started


@pytest.mark.asyncio
async def test_goal_proposal_none_means_no_gate(tmp_path):
    client = GatedClient()
    client.add([TextChunk(
        text="<verify_proposal><none/></verify_proposal> 这是审美类目标"
    )])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await type_and_submit(pilot, input_widget, "/goal 让代码更优雅")
        await wait_until(
            pilot,
            lambda: "没有客观可判定的验收方式" in chat.transcript_text(),
        )
        # No picker popped for an explicit <none/>.
        assert isinstance(pilot.app.screen_stack[-1], MainScreen)


@pytest.mark.asyncio
async def test_goal_badge_rainbow_while_running(tmp_path, monkeypatch):
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="still working")], gate=gate)

    async def passing(command, workdir, **kwargs):
        return VerifyResult(exit_code=0)

    monkeypatch.setattr(goal_driver_module, "run_verify", passing)

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
async def test_goal_edit_mode_accepts_edited_command(tmp_path, monkeypatch):
    """M2 edit path: picker → 自己编辑 → prefill → submit → accepted."""
    client = GatedClient()
    client.add([TextChunk(
        text="<verify_proposal><command>pytest -x</command></verify_proposal>"
    )])
    client.add([TextChunk(text="working")])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        async def passing(command, workdir, **kwargs):
            return VerifyResult(exit_code=0)

        monkeypatch.setattr(goal_driver_module, "run_verify", passing)

        await type_and_submit(pilot, input_widget, "/goal 补齐测试")
        await wait_picker(pilot)
        # Move to "✏️ 自己输入/编辑命令…" (second row) and select it.
        await pilot.press("down")
        await pilot.press("enter")
        await wait_main(pilot)
        # Input is prefilled with the best proposal; edit and submit.
        assert input_widget.text == "pytest -x"
        await type_and_submit(pilot, input_widget, "pytest tests/tools -q")
        await wait_until(pilot, lambda: "目标达成" in chat.transcript_text())
        state = screen.driver.status()
        assert state is not None
        assert state.verify_command == "pytest tests/tools -q"
