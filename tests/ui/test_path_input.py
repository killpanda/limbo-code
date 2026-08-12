"""'/'-leading paths: command fallback, auto-attach, escape hatch, menu hints."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.path_input import attachments_from_text, looks_like_path
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.command_menu import SlashCommandMenu
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


def test_looks_like_path_multi_segment():
    assert looks_like_path("/Users/admin/docs.pdf")
    # Missing files still count: the shape is unambiguous.
    assert looks_like_path("/no/such/file.txt")


def test_looks_like_path_existing_single_segment(tmp_path):
    assert looks_like_path(str(tmp_path))


def test_command_names_are_not_paths():
    # Single segment, does not resolve on disk.
    assert not looks_like_path("/hlep")
    assert not looks_like_path("/goal")


# -- attachments_from_text -------------------------------------------------------


def test_attachments_from_text_existing_file(tmp_path):
    doc = tmp_path / "docs.txt"
    doc.write_text("hello", encoding="utf-8")
    attachments = attachments_from_text(str(doc))
    assert len(attachments) == 1
    assert attachments[0].kind == "file"
    assert attachments[0].name == "docs.txt"
    assert attachments[0].path == str(doc)


def test_attachments_from_text_image_sniffed_by_magic_bytes(tmp_path):
    img = tmp_path / "pic.txt"  # mis-named on purpose: content wins
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    (attachment,) = attachments_from_text(str(img))
    assert attachment.kind == "image"
    assert attachment.mime == "image/png"


def test_attachments_from_text_skips_missing_dirs_relative_and_dupes(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("x", encoding="utf-8")
    text = f"{tmp_path}/missing.txt {tmp_path} docs/rel.txt {doc} {doc}"
    attachments = attachments_from_text(text)
    assert [a.path for a in attachments] == [str(doc)]


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


@pytest.mark.asyncio
async def test_missing_path_like_text_sends_plain_message(tmp_path):
    fake_llm = FakeLLMClient([[TextChunk(text="done")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, "/no/such/file.pdf")
        await wait_until(pilot, lambda: "done" in chat.transcript_text())

        transcript = chat.transcript_text()
        assert "未知命令" not in transcript
        assert "📎" not in transcript
        assert fake_llm.calls[0][-1].content == "/no/such/file.pdf"


@pytest.mark.asyncio
async def test_image_path_auto_attaches_as_image(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    fake_llm = FakeLLMClient([[TextChunk(text="done")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, str(img))
        await wait_until(pilot, lambda: "done" in chat.transcript_text())

        assert "📎 图片: shot.png" in chat.transcript_text()


# -- unknown command error + escape hatch (C) -------------------------------------


@pytest.mark.asyncio
async def test_unknown_command_error_points_to_escape_hatch(tmp_path):
    fake_llm = FakeLLMClient()
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, "/hlep")
        await pilot.pause()

        transcript = chat.transcript_text()
        assert "未知命令 /hlep" in transcript
        assert "加一个空格" in transcript
        assert not fake_llm.calls


@pytest.mark.asyncio
async def test_leading_space_forces_plain_text(tmp_path):
    fake_llm = FakeLLMClient([[TextChunk(text="done")]])
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        await submit(pilot, input_widget, " /hlep")
        await wait_until(pilot, lambda: "done" in chat.transcript_text())

        assert "未知命令" not in chat.transcript_text()
        assert fake_llm.calls[0][-1].content == "/hlep"


# -- typing hints (D) --------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_match_shows_passive_hint(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        menu = screen.query_one("#slash-menu", SlashCommandMenu)

        await pilot.press("/", "x")
        await pilot.pause()

        assert menu.is_open
        assert not screen.slash_menu_open  # passive: keys keep normal behavior
        prompts = [str(o.prompt) for o in menu.options]
        assert prompts == ["无匹配命令 · /help 查看全部命令"]

        # Enter submits normally (unknown command error), hint closes.
        await pilot.press("enter")
        await pilot.pause()
        assert not menu.is_open


@pytest.mark.asyncio
async def test_path_hints_while_typing(tmp_path):
    doc = tmp_path / "docs.txt"
    doc.write_text("hello", encoding="utf-8")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        menu = screen.query_one("#slash-menu", SlashCommandMenu)
        input_widget = screen.query_one("#input", InputWidget)

        # Missing path-like input: plain-text hint.
        input_widget.text = "/no/such/file.pdf"
        await pilot.pause()
        prompts = [str(o.prompt) for o in menu.options]
        assert prompts == ["无匹配命令 · 将作为普通文本发送"]

        # Existing file: attachment hint.
        input_widget.text = str(doc)
        await pilot.pause()
        prompts = [str(o.prompt) for o in menu.options]
        assert prompts == ["📎 路径已识别 · 回车将作为附件发送"]


@pytest.mark.asyncio
async def test_matching_prefix_still_shows_commands_not_hint(tmp_path):
    """Regression: partial command matches keep the interactive menu."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        menu = screen.query_one("#slash-menu", SlashCommandMenu)

        await pilot.press("/", "h", "e")
        await pilot.pause()

        assert menu.is_open
        assert screen.slash_menu_open
        prompts = [str(o.prompt) for o in menu.options]
        assert any("/help" in p for p in prompts)
