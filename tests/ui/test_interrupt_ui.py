"""UI tests for ESC turn interruption (RFC LIM-53).

Covers the RFC Test Plan pilot layer: the ESC routing chain (verify →
interrupt turn → cancel queued → no-op), the interrupted-turn feedback
(cancelled tool card, "⏹ 已打断", statusbar back to idle), and the
no-auto-follow-up rule for queued steers.
"""

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


class GateMidStreamClient:
    """Yields a tool call, then parks mid-stream until released."""

    def __init__(self):
        self.calls = []
        self.gate = asyncio.Event()

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))
        yield ToolCallEvent(
            id="t1", name="bash", arguments={"command": "sleep 5"}
        )
        await self.gate.wait()
        yield TextChunk(text="tail")


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


@pytest.mark.asyncio
async def test_esc_interrupt_marks_partial_tool_card_cancelled(tmp_path):
    """A tool call seen only mid-stream never executes: its card must not
    stay in the running state after the interrupt."""
    client = GateMidStreamClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: "t1" in chat.tool_cards)
        assert chat.tool_cards["t1"].state == "running"

        await pilot.press("escape")
        await wait_until(pilot, lambda: not screen.driver.running)

        assert chat.tool_cards["t1"].state == "cancelled"
        assert "⏹ 已打断" in chat.transcript_text()
        assert "● idle" in str(statusbar._state_label.content)
        # The partial call was dropped from history (content-only append
        # with nothing streamed → no assistant message at all).
        assert screen.agent.messages[-1].role == "user"
        assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_esc_routes_to_verify_before_interrupt(tmp_path):
    """While a verify subprocess is in flight, ESC cancels the verify
    (LIM-40) instead of reaching the turn-interrupt layer."""
    client = GateMidStreamClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)

        # Fake an in-flight verify subprocess (white-box: GoalDriver owns it).
        verify_task = asyncio.create_task(asyncio.sleep(60))
        screen.driver._verify_task = verify_task

        await pilot.press("escape")
        await pilot.pause()

        assert verify_task.cancelled()
        assert "已取消验收命令" in chat.transcript_text()
        assert "⏹ 已打断" not in chat.transcript_text()


@pytest.mark.asyncio
async def test_esc_interrupts_when_input_not_focused(tmp_path):
    """马越 acceptance feedback: ESC must work when the input widget does
    not have focus (screen-level binding routes through the same chain)."""
    client = GateMidStreamClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: "t1" in chat.tool_cards)

        # Move focus off the input (e.g. the user clicked the chat flow).
        chat.focus()
        await pilot.pause()
        assert screen.app.focused is not input_widget

        await pilot.press("escape")
        await wait_until(pilot, lambda: not screen.driver.running)

        assert chat.tool_cards["t1"].state == "cancelled"
        assert "⏹ 已打断" in chat.transcript_text()
        assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_esc_unfocused_falls_back_to_queue_cancel(tmp_path):
    """Unfocused ESC with no running turn still cancels the newest queued
    steer (the fallback layer keeps working from any focus)."""
    client = GateMidStreamClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        input_widget.text = "排队消息"
        await pilot.press("enter")
        await pilot.pause()
        assert screen.agent.queued_count == 1

        # Interrupt the turn first (unfocused).
        chat.focus()
        await pilot.pause()
        await pilot.press("escape")
        await wait_until(pilot, lambda: not screen.driver.running)
        assert screen.agent.queued_count == 1

        # The turn's finally-chain hands focus back to the input; move it
        # off again (user clicks the chat after the turn ends), then ESC
        # still reaches the queue-cancel fallback.
        chat.focus()
        await pilot.pause()
        assert screen.app.focused is not input_widget
        await pilot.press("escape")
        await pilot.pause()
        assert screen.agent.queued_count == 0
        assert "❯ 排队消息（已取消）" in chat.transcript_text()


@pytest.mark.asyncio
async def test_esc_noop_when_idle_and_queue_empty(tmp_path):
    client = GateMidStreamClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        before = chat.transcript_text()

        await pilot.press("escape")
        await pilot.pause()

        assert chat.transcript_text() == before
        assert not screen.driver.running
