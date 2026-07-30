"""Tests for InputWidget clipboard attachment pasting (RFC LIM-17 P2)."""

from pathlib import Path

import pytest
from textual.app import App
from textual.widgets import TextArea

from limbo.ui import clipboard
from limbo.ui.widgets.input import InputWidget, UserSubmitted


def make_app(submitted: list | None = None):
    class TestApp(App[None]):
        def compose(self):
            yield InputWidget(id="input")

        def on_user_submitted(self, event: UserSubmitted) -> None:
            if submitted is not None:
                submitted.append(event)

    return TestApp()


@pytest.mark.asyncio
async def test_ctrl_v_image_clipboard_creates_attachment(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        clipboard,
        "read_clipboard",
        lambda: clipboard.ClipboardImage(b"png-bytes", "png"),
    )
    monkeypatch.setattr(clipboard, "ATTACHMENTS_DIR", tmp_path)
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await pilot.press("ctrl+v")
        await pilot.pause()
        assert widget.text == "[图片 #1]"
        assert len(widget._attachments) == 1
        _, attachment = widget._attachments[0]
        assert attachment.kind == "image"
        assert attachment.mime == "image/png"
        assert Path(attachment.path).read_bytes() == b"png-bytes"

        widget.action_submit()
        await pilot.pause()
        assert len(submitted) == 1
        assert submitted[0].message == "[图片 #1]"
        assert len(submitted[0].attachments) == 1
        assert submitted[0].attachments[0].kind == "image"
        # clear() resets attachment state for the next message.
        assert widget._attachments == []


@pytest.mark.asyncio
async def test_ctrl_v_file_clipboard_creates_attachment(
    monkeypatch, tmp_path: Path
):
    target = tmp_path / "report.txt"
    target.write_text("data")
    monkeypatch.setattr(
        clipboard, "read_clipboard", lambda: clipboard.ClipboardFiles([target])
    )
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await pilot.press("ctrl+v")
        await pilot.pause()
        assert widget.text == "[文件 #1: report.txt]"
        _, attachment = widget._attachments[0]
        assert attachment.kind == "file"
        assert attachment.path == str(target)


@pytest.mark.asyncio
async def test_ctrl_v_text_clipboard_falls_back_to_builtin_paste(monkeypatch):
    """Text/empty clipboard must defer to TextArea's original paste path."""
    monkeypatch.setattr(
        clipboard, "read_clipboard", lambda: clipboard.ClipboardText("t")
    )
    called = []
    original = TextArea.action_paste

    def traced(self):
        called.append(True)
        return original(self)

    monkeypatch.setattr(TextArea, "action_paste", traced)
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await pilot.press("ctrl+v")
        await pilot.pause()
        assert called == [True]
        assert widget._attachments == []


@pytest.mark.asyncio
async def test_ctrl_v_empty_clipboard_falls_back(monkeypatch):
    monkeypatch.setattr(
        clipboard, "read_clipboard", lambda: clipboard.ClipboardEmpty()
    )
    called = []
    monkeypatch.setattr(
        TextArea, "action_paste", lambda self: called.append(True)
    )
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await pilot.press("ctrl+v")
        await pilot.pause()
        assert called == [True]


@pytest.mark.asyncio
async def test_submit_drops_attachments_whose_marker_was_deleted(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        clipboard,
        "read_clipboard",
        lambda: clipboard.ClipboardImage(b"x", "png"),
    )
    # ctrl+v persists the image — keep it out of the real ~/.limbo.
    monkeypatch.setattr(clipboard, "ATTACHMENTS_DIR", tmp_path / "attachments")
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await pilot.press("ctrl+v")
        await pilot.pause()
        # Delete the marker by hand, keep other text.
        widget.text = "hello"
        widget.action_submit()
        await pilot.pause()
        assert submitted[0].attachments == []


@pytest.mark.asyncio
async def test_ctrl_v_attachment_marker_replaces_selection(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        clipboard,
        "read_clipboard",
        lambda: clipboard.ClipboardImage(b"png-bytes", "png"),
    )
    monkeypatch.setattr(clipboard, "ATTACHMENTS_DIR", tmp_path)
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "hello world"
        widget.selection = ((0, 6), (0, 11))
        await pilot.press("ctrl+v")
        await pilot.pause()
        await pilot.pause()
        assert widget.text == "hello [图片 #1]"
        assert "world" not in widget.text
