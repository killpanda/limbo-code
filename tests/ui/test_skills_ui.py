"""Tests for skill support in the slash-command menu and invocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))  # snapshot; the agent mutates its list
        if self.responses:
            for event in self.responses.pop(0):
                yield event


def make_app(tmp_path: Path, fake_llm=None) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm or FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


def write_project_skill(
    workdir: Path, name: str, description: str, body: str
) -> None:
    skill_dir = workdir / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


async def wait_for(pilot, needle: str, app) -> None:
    chat = app.screen.query_one("#chat", ChatWidget)
    input_widget = app.screen.query_one("#input", InputWidget)
    for _ in range(200):
        await pilot.pause()
        if needle in chat.transcript_text() and not input_widget.disabled:
            return
    raise AssertionError(f"timed out waiting for {needle!r}")


@pytest.mark.asyncio
async def test_invoking_skill_sends_body_and_args_to_llm(tmp_path):
    write_project_skill(tmp_path, "tdd", "Test-driven development.", "# TDD rules\nRed first.")
    fake_llm = FakeLLMClient([[TextChunk(text="skill loaded")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        input_widget = app.screen.query_one("#input", InputWidget)
        input_widget.text = "/tdd implement login"
        await pilot.press("enter")
        await wait_for(pilot, "skill loaded", app)

        assert len(fake_llm.calls) == 1
        prompt = fake_llm.calls[0][-1].content  # last user message
        assert "# TDD rules" in prompt
        assert "Red first." in prompt
        assert "implement login" in prompt
        # The chat shows the compact invocation, not the whole skill body.
        chat = app.screen.query_one("#chat", ChatWidget)
        assert "❯ /tdd implement login" in chat.transcript_text()


