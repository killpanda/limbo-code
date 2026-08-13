"""'!' bang commands: direct bash execution, tool-card rendering, isolation.

RFC: design/rfc-bang-command.md
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from limbo.agent import InterruptEvent
from limbo.config import Config
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget
from limbo.ui.widgets.tool_card import ToolCard


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[list] = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))
        if self.responses:
            for event in self.responses.pop(0):
                yield event


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


async def wait_until(pilot, predicate, attempts: int = 400) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError("timed out waiting for condition")


async def submit(pilot, input_widget: InputWidget, text: str) -> None:
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


def bang_card(chat: ChatWidget) -> ToolCard | None:
    for card in chat.tool_cards.values():
        if card.tool_name == "bash":
            return card
    return None


async def wait_for_bang(pilot, chat: ChatWidget) -> ToolCard:
    await wait_until(
        pilot,
        lambda: (card := bang_card(chat)) is not None and card.state != "running",
    )
    card = bang_card(chat)
    assert card is not None
    return card


# -- Test Plan #1: bang executes locally, never touches the LLM ----------------


@pytest.mark.asyncio
async def test_bang_runs_without_llm(tmp_path):
    client = FakeLLMClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, "!echo hello")
        card = await wait_for_bang(pilot, chat)

        assert card.state == "success"
        assert card._body_content is not None
        assert "hello" in card._body_content[0]
        # No LLM request, no history growth (system message only).
        assert client.calls == []
        assert len(screen.agent.messages) == 1
        # Echoed as a user message, without attachments.
        assert "❯ !echo hello" in chat.transcript_text()


# -- Test Plan #2: non-zero exit code → error card with full output ------------


@pytest.mark.asyncio
async def test_bang_nonzero_exit_shows_error_card(tmp_path):
    app = make_app(tmp_path, FakeLLMClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, "!echo oops && exit 3")
        card = await wait_for_bang(pilot, chat)

        assert card.state == "error"
        assert card._body_content is not None
        assert "exit code 3" in card._body_content[0]
        assert "oops" in card._body_content[0]


# -- Test Plan #3: bare '!' shows usage, executes nothing ----------------------


@pytest.mark.asyncio
async def test_bare_bang_shows_usage(tmp_path):
    client = FakeLLMClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, "!")
        await wait_until(pilot, lambda: "用法：!" in chat.transcript_text())

        assert bang_card(chat) is None
        assert client.calls == []
        assert len(screen.agent.messages) == 1


# -- Test Plan #4: leading space escapes to a plain LLM message ----------------


@pytest.mark.asyncio
async def test_leading_space_forces_plain_text(tmp_path):
    client = FakeLLMClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, " !echo hi")
        await wait_until(pilot, lambda: len(client.calls) == 1)

        assert client.calls[0][-1].content == "!echo hi"
        assert bang_card(chat) is None


# -- Test Plan #5: bang while busy runs in parallel, never steers --------------


@pytest.mark.asyncio
async def test_bang_while_busy_runs_in_parallel(tmp_path):
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="working")], gate=gate)
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        # Spy on the agent registry's bash tool: bang must use the screen's
        # private instance, never the agent's.
        agent_bash = screen.agent.registry.get("bash")
        assert agent_bash is not None
        agent_calls = []
        original_execute = agent_bash.execute

        def spy(arguments):
            agent_calls.append(arguments)
            return original_execute(arguments)

        agent_bash.execute = spy  # type: ignore[method-assign]

        screen.run_worker(screen._handle_turn("hi"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        assert screen._agent_busy

        await submit(pilot, input_widget, "!echo parallel")
        card = await wait_for_bang(pilot, chat)

        assert card.state == "success"
        assert card._body_content is not None
        assert "parallel" in card._body_content[0]
        # Not queued as steer; the agent's bash tool was never touched.
        assert screen.agent.queued_count == 0
        assert agent_calls == []

        gate.set()
        await wait_until(pilot, lambda: not screen._agent_busy)


# -- Test Plan #6 (B1): turn interrupt must not cancel bang cards ---------------


@pytest.mark.asyncio
async def test_interrupt_does_not_cancel_bang_card(tmp_path):
    app = make_app(tmp_path, FakeLLMClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        # An agent-owned card still waiting for its tool result.
        agent_card = chat.add_tool_card("c1", "read", {"path": "x.py"})
        assert agent_card.state == "running"

        await submit(pilot, input_widget, "!sleep 0.3 && echo bang-done")
        await wait_until(pilot, lambda: bang_card(chat) is not None)
        card = bang_card(chat)
        assert card is not None and card.state == "running"

        # The turn-interrupt path: InterruptEvent → cancel_running_tool_cards.
        await screen._process_agent_event(InterruptEvent(phase="llm_stream"))

        assert agent_card.state == "cancelled"
        assert card.state == "running"  # bang keeps running, result arrives

        await wait_until(pilot, lambda: card.state != "running")
        assert card.state == "success"
        assert card._body_content is not None
        assert "bang-done" in card._body_content[0]


@pytest.mark.asyncio
async def test_cancel_running_tool_cards_skips_non_agent_owned(tmp_path):
    """Widget-level unit: only agent-owned running cards are cancelled."""
    app = make_app(tmp_path, FakeLLMClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)

        agent_card = chat.add_tool_card("a1", "read", {"path": "x"})
        user_card = chat.add_tool_card(
            "bang-1", "bash", {"command": "ls"}, agent_owned=False
        )
        done_card = chat.add_tool_card("a2", "ls", {})
        done_card.set_success("ok")

        chat.cancel_running_tool_cards()

        assert agent_card.state == "cancelled"
        assert user_card.state == "running"
        assert done_card.state == "success"


# -- Test Plan #7: _editing_verify mode treats '!cmd' as the verify command ----


@pytest.mark.asyncio
async def test_editing_verify_treats_bang_text_literally(tmp_path):
    client = FakeLLMClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.pump.set_goal("make tests pass")
        screen._editing_verify = True
        await submit(pilot, input_widget, "!pytest -x")
        await pilot.pause()

        state = screen.pump.status()
        assert state is not None
        assert state.verify_command == "!pytest -x"
        assert bang_card(chat) is None
        assert not screen._editing_verify


# -- Test Plan #8: worker fallback turns unexpected exceptions into errors -----


@pytest.mark.asyncio
async def test_bang_worker_exception_renders_error_card(tmp_path, monkeypatch):
    app = make_app(tmp_path, FakeLLMClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        def boom(arguments):
            raise RuntimeError("boom")

        monkeypatch.setattr(screen._bang_tool, "execute", boom)

        await submit(pilot, input_widget, "!echo hi")
        card = await wait_for_bang(pilot, chat)

        assert card.state == "error"
        assert card._body_content is not None
        assert "内部错误：boom" in card._body_content[0]


# -- Test Plan #9: bang commands join the input history -------------------------


@pytest.mark.asyncio
async def test_bang_joins_input_history(tmp_path):
    app = make_app(tmp_path, FakeLLMClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, "!ls")
        await wait_for_bang(pilot, chat)

        await pilot.press("up")
        await pilot.pause()
        assert input_widget.text == "!ls"
