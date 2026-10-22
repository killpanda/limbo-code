"""'/'-leading paths: command fallback, auto-attach, escape hatch, menu hints."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[list] = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))
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


def get_screen(pilot) -> MainScreen:
    return pilot.app.screen_stack[-1]


async def wait_until(pilot, predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError("timed out waiting for condition")


async def submit(pilot, input_widget: InputWidget, text: str) -> None:
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


# -- looks_like_path ------------------------------------------------------------


# -- attachments_from_text -------------------------------------------------------


# -- submit fallback (A) + auto-attach (B) ---------------------------------------


@pytest.mark.asyncio
async def test_existing_path_submits_as_message_with_attachment(tmp_path):
    doc = tmp_path / "docs.txt"
    doc.write_text("file body here", encoding="utf-8")
    fake_llm = FakeLLMClient([[TextChunk(text="done")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, str(doc))
        await wait_until(pilot, lambda: "done" in chat.transcript_text())

        transcript = chat.transcript_text()
        assert "未知命令" not in transcript
        assert "📎 文件: docs.txt" in transcript  # attachment chip
        assert fake_llm.calls, "path must reach the model as a message"
        content = fake_llm.calls[0][-1].content
        assert str(doc) in content
        assert "file body here" in content  # small text file inlined


# -- unknown command error + escape hatch (C) -------------------------------------


# -- typing hints (D) --------------------------------------------------------------


