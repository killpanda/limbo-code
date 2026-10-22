"""'!' bang commands: direct bash execution, tool-card rendering, isolation.

RFC: design/rfc-bang-command.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
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
        assert "hello" in card._body_content[0][0]
        # No LLM request, no history growth (system message only).
        assert client.calls == []
        assert len(screen.agent.messages) == 1
        # Echoed as a user message, without attachments.
        assert "❯ !echo hello" in chat.transcript_text()


# -- Test Plan #2: non-zero exit code → error card with full output ------------


# -- Test Plan #3: bare '!' shows usage, executes nothing ----------------------


# -- Test Plan #4: leading space escapes to a plain LLM message ----------------


# -- Test Plan #5: bang while busy runs in parallel, never steers --------------


# -- Test Plan #6 (B1): turn interrupt must not cancel bang cards ---------------


# -- Test Plan #7: _editing_verify mode treats '!cmd' as the verify command ----


# -- Test Plan #8: worker fallback turns unexpected exceptions into errors -----


# -- Test Plan #9: bang commands join the input history -------------------------


