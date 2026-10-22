"""UI tests for /goal: command mounting, busy semantics, badge (LIM-40)."""

from __future__ import annotations

from pathlib import Path

import pytest

import limbo.pump as pump_module
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

        monkeypatch.setattr(pump_module, "run_verify", passing)

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
        state = screen.pump.status()
        assert state is not None and state.status == "passed"
        assert state.verify_command == "python3 -m unittest discover -v"
        # First turn is the proposal round (proposer prompt, not work prompt).
        assert "先不要动手" in client.calls[0][-1].content


