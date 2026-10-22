"""Tests for mid-turn steer messaging UI (RFC LIM-20 + v2.1 cancel)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent
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


def write_project_skill(workdir: Path, name: str, description: str, body: str) -> None:
    skill_dir = workdir / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


# -- queueing and delivery (Test Plan #12, #13, #19) ---------------------------


@pytest.mark.asyncio
async def test_submit_while_busy_queues_and_delivers(tmp_path):
    """Busy: input stays usable, Enter queues (no second worker), the card
    flips on SteerEvent, the status bar counts down to zero."""
    gate = asyncio.Event()
    client = GatedClient()
    client.add([ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})])
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])
    (tmp_path / "main.py").write_text("x = 1\n")
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)

        screen.run_worker(screen._handle_turn("read main.py"))
        await wait_until(pilot, lambda: len(client.calls) == 2)

        # Input stays usable mid-turn; no second LLM call on submit.
        assert screen._agent_busy
        assert not input_widget.disabled
        await type_and_submit(pilot, input_widget, "改用 uv")
        assert len(client.calls) == 2
        assert screen.agent.queued_count == 1
        assert "❯ 改用 uv（排队中）" in chat.transcript_text()
        assert "1 条排队中" in statusbar._context_text()

        gate.set()
        await wait_until(
            pilot, lambda: len(client.calls) == 3 and "done" in chat.transcript_text()
        )

        # Injected before the next request; card flipped; counter cleared.
        assert client.calls[2][-1].content == "改用 uv"
        transcript = chat.transcript_text()
        assert "❯ 改用 uv" in transcript
        assert "（排队中）" not in transcript
        assert "排队中" not in statusbar._context_text()
        assert not screen._agent_busy


# -- busy command guards (Test Plan #14, #15) ------------------------------------


# -- skill queueing (Test Plan #16) ----------------------------------------------


# -- error-exit auto follow-up (Test Plan #17) -----------------------------------


# -- attachments (Test Plan #18) ---------------------------------------------------


# -- cancel: Esc (Test Plan #25, #26, #30) ----------------------------------------


# -- cancel: card ✕ (Test Plan #27, #28) ------------------------------------------


# -- cancel: skill prompt (Test Plan #29) ------------------------------------------


# -- interaction with LIM-19 queued steer --------------------------------------


