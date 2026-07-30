"""Tests for the cross-platform clipboard reader (backends are mocked)."""

from __future__ import annotations

import base64

from limbo.ui import clipboard
from limbo.ui.clipboard import (
    ClipboardEmpty,
    ClipboardFiles,
    ClipboardImage,
    ClipboardText,
)


def _scripted_run(responses: dict[tuple, str | bytes | None]):
    """A _run stand-in keyed by the argument tuple."""

    def fake_run(args):
        value = responses.get(tuple(args))
        if value is None:
            return None

        class Result:
            stdout = value.encode() if isinstance(value, str) else value

        return Result()

    return fake_run


def _mock_run(monkeypatch, responses: dict[tuple, str | bytes | None]):
    monkeypatch.setattr(clipboard, "_run", _scripted_run(responses))


# -- macOS ----------------------------------------------------------------------


def test_macos_image_clipboard(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    png = b"\x89PNG\r\n\x1a\nfake"
    hex_dump = "«data PNGf" + png.hex().upper() + "»"
    _mock_run(
        monkeypatch,
        {
            ("osascript", "-e", "clipboard info"): "«class PNGf», 1234",
            (
                "osascript",
                "-e",
                "set theData to the clipboard as «class PNGf»",
                "-e",
                "return theData",
            ): hex_dump,
        },
    )
    result = clipboard.read_clipboard()
    assert isinstance(result, ClipboardImage)
    assert result.data == png
    assert result.ext == "png"


def test_macos_text_clipboard(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    _mock_run(
        monkeypatch,
        {
            ("osascript", "-e", "clipboard info"): "«class utf8», 5",
            ("pbpaste",): "hello",
        },
    )
    assert clipboard.read_clipboard() == ClipboardText("hello")


def test_macos_empty_clipboard(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    _mock_run(
        monkeypatch,
        {("osascript", "-e", "clipboard info"): "", ("pbpaste",): ""},
    )
    assert clipboard.read_clipboard() == ClipboardEmpty()


def test_macos_tool_failure_degrades(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(clipboard, "_run", lambda args: None)
    assert clipboard.read_clipboard() == ClipboardEmpty()


# -- Linux -----------------------------------------------------------------------


def test_linux_wl_paste_image(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setattr(clipboard.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    _mock_run(
        monkeypatch,
        {
            ("wl-paste", "--list-types"): "image/png\ntext/plain",
            ("wl-paste", "--type", "image/png"): b"png-bytes",
        },
    )
    result = clipboard.read_clipboard()
    assert isinstance(result, ClipboardImage)
    assert result.data == b"png-bytes"
    assert result.ext == "png"


def test_linux_wl_paste_uri_list(monkeypatch, tmp_path):
    target = tmp_path / "report.txt"
    target.write_text("data")
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.setattr(clipboard.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    _mock_run(
        monkeypatch,
        {
            ("wl-paste", "--list-types"): "text/uri-list",
            ("wl-paste", "--type", "text/uri-list"): f"file://{target}\n",
        },
    )
    assert clipboard.read_clipboard() == ClipboardFiles([target])


def test_linux_xclip_text(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None,
    )
    _mock_run(
        monkeypatch,
        {
            ("xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"): "UTF8_STRING",
            ("xclip", "-selection", "clipboard", "-o"): "plain text",
        },
    )
    assert clipboard.read_clipboard() == ClipboardText("plain text")


def test_linux_no_clipboard_tool(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(clipboard.shutil, "which", lambda cmd: None)
    assert clipboard.read_clipboard() == ClipboardEmpty()


# -- Windows ---------------------------------------------------------------------


def test_windows_image_clipboard(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Windows")
    png = b"png-bytes"
    outputs = iter(["image", base64.b64encode(png).decode()])
    monkeypatch.setattr(clipboard, "_powershell", lambda script: next(outputs))
    result = clipboard.read_clipboard()
    assert isinstance(result, ClipboardImage)
    assert result.data == png


def test_windows_files_clipboard(monkeypatch, tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x")
    outputs = iter(["files", f"{target}\n"])
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Windows")
    monkeypatch.setattr(clipboard, "_powershell", lambda script: next(outputs))
    assert clipboard.read_clipboard() == ClipboardFiles([target])


def test_windows_text_clipboard(monkeypatch):
    outputs = iter(["text", "hello\r\n"])
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Windows")
    monkeypatch.setattr(clipboard, "_powershell", lambda script: next(outputs))
    assert clipboard.read_clipboard() == ClipboardText("hello")


# -- misc ------------------------------------------------------------------------


def test_unsupported_platform(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Plan9")
    assert clipboard.read_clipboard() == ClipboardEmpty()


def test_save_clipboard_image(monkeypatch, tmp_path):
    monkeypatch.setattr(clipboard, "ATTACHMENTS_DIR", tmp_path)
    path = clipboard.save_clipboard_image(b"bytes", "png")
    assert path.read_bytes() == b"bytes"
    assert path.parent == tmp_path


def test_paths_from_uri_list(tmp_path):
    good = tmp_path / "exists.txt"
    good.write_text("x")
    uri_list = f"# comment\nfile://{good}\nfile://{tmp_path}/missing.txt\nhttps://x\n"
    assert clipboard._paths_from_uri_list(uri_list) == [good]


def test_uri_list_unquotes_spaces(tmp_path):
    spaced = tmp_path / "my file.txt"
    spaced.write_text("x")
    uri = f"file://{tmp_path}/my%20file.txt"
    assert clipboard._paths_from_uri_list(uri) == [spaced]
