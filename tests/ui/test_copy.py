"""Copy-to-clipboard tests (LIM-54).

Root cause on macOS Terminal.app: Textual's default copy path only emits an
OSC 52 escape (silently ignored by Terminal.app), and Cmd+C never reaches
the app (Terminal.app handles it natively with an empty native selection,
since the app captured the mouse). The fix routes clipboard writes through
native tools (``pbcopy`` etc.) and auto-copies on selection end.
"""

from __future__ import annotations

import pytest

from limbo.ui import clipboard
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget

# -- write_clipboard_text -------------------------------------------------------


def test_write_macos_uses_pbcopy(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    # CI runners are Linux without pbcopy; _write_cmd short-circuits on
    # shutil.which, so it must be mocked too.
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: "/usr/bin/pbcopy")
    calls: list[tuple[list[str], bytes]] = []

    def fake_run(args, input=None, **kw):
        calls.append((args, input))

        class Result:
            returncode = 0
            stdout = b""

        return Result()

    monkeypatch.setattr(clipboard.subprocess, "run", fake_run)
    assert clipboard.write_clipboard_text("hello") is True
    assert calls == [(["pbcopy"], b"hello")]


def test_write_returns_false_when_tool_missing(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: None)
    assert clipboard.write_clipboard_text("hello") is False


def test_write_empty_text_is_noop(monkeypatch):
    def boom(*a, **kw):  # pragma: no cover - must never be called
        raise AssertionError("subprocess.run must not run for empty text")

    monkeypatch.setattr(clipboard.subprocess, "run", boom)
    assert clipboard.write_clipboard_text("") is False


def test_write_backend_error_returns_false(monkeypatch):
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    # Keep which() present so the failure is exercised at returncode=1,
    # not at the missing-tool short-circuit.
    monkeypatch.setattr(clipboard.shutil, "which", lambda _: "/usr/bin/pbcopy")

    class Result:
        returncode = 1
        stdout = b""

    monkeypatch.setattr(clipboard.subprocess, "run", lambda *a, **kw: Result())
    assert clipboard.write_clipboard_text("hello") is False


# -- App integration -------------------------------------------------------------


def _native_stub(written: list[str], ok: bool = True):
    """A write_clipboard_text stand-in that records calls (thread-safe enough
    for the single writer these tests produce)."""

    def stub(text: str) -> bool:
        written.append(text)
        return ok

    return stub


async def _drain_clipboard_worker(pilot) -> None:
    """Let the off-thread clipboard write finish before asserting."""
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


async def _wait_laid_out(pilot, widget) -> None:
    """Wait until the widget has a real layout region.

    A single ``pilot.pause()`` after adding a message does not guarantee
    the layout pass has run on a cold app (first app of the pytest
    process, slow CI runner): the drag then lands on a zero-sized widget
    and silently selects nothing.
    """
    for _ in range(50):
        await pilot.pause()
        if widget.region.width > 0 and widget.region.height > 0:
            return
    raise AssertionError("widget was never laid out")

# The startup banner art occupies the top rows; chat messages render below it.
# A drag across row 3 of a fresh app selects banner art, which is fine for
# mechanism tests — what matters is that *something* selectable lands in the
# clipboard.


@pytest.mark.asyncio
async def test_copy_to_clipboard_prefers_native_tool(monkeypatch, tmp_path):
    """copy_to_clipboard must not rely on OSC 52 alone (Terminal.app drops it)."""
    written: list[str] = []
    monkeypatch.setattr("limbo.ui.app.write_clipboard_text", _native_stub(written))
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        app.copy_to_clipboard("native-wins")
        await _drain_clipboard_worker(pilot)
    assert written == ["native-wins"]


@pytest.mark.asyncio
async def test_copy_to_clipboard_keeps_app_clipboard_in_sync(monkeypatch, tmp_path):
    """App.clipboard must reflect the text even when the native backend wins."""
    written: list[str] = []
    monkeypatch.setattr("limbo.ui.app.write_clipboard_text", _native_stub(written))
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        app.copy_to_clipboard("in-sync")
        await _drain_clipboard_worker(pilot)
    assert app.clipboard == "in-sync"


@pytest.mark.asyncio
async def test_native_write_runs_off_event_loop_thread(monkeypatch, tmp_path):
    """Regression (review): the subprocess write must not block the UI loop."""
    import threading

    threads: list[int] = []

    def stub(text: str) -> bool:
        threads.append(threading.get_ident())
        return True

    monkeypatch.setattr("limbo.ui.app.write_clipboard_text", stub)
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        app.copy_to_clipboard("off-thread")
        await _drain_clipboard_worker(pilot)
    assert threads and threads[0] != threading.main_thread().ident


@pytest.mark.asyncio
async def test_copy_to_clipboard_falls_back_to_osc52(monkeypatch, tmp_path):
    """Without a native backend (e.g. SSH), OSC 52 is still emitted."""
    written: list[str] = []
    monkeypatch.setattr(
        "limbo.ui.app.write_clipboard_text", _native_stub(written, ok=False)
    )
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        osc52: list[str] = []
        driver = app._driver
        real_write = driver.write

        def spy_write(data: str) -> None:
            if "\x1b]52;" in data:
                osc52.append(data)
            real_write(data)

        monkeypatch.setattr(driver, "write", spy_write)
        app.copy_to_clipboard("fallback")
        await _drain_clipboard_worker(pilot)
    assert len(osc52) == 1


@pytest.mark.asyncio
async def test_drag_selection_auto_copies_on_mouse_up(monkeypatch, tmp_path):
    """Releasing the mouse after a drag-select copies the selected text.

    This is what makes select-then-Cmd+C work on Terminal.app: the text is
    already on the system clipboard before Terminal.app's native (no-op)
    copy runs.
    """
    written: list[str] = []
    monkeypatch.setattr("limbo.ui.app.write_clipboard_text", _native_stub(written))
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        chat = app.screen.query_one(ChatWidget)
        chat.add_assistant_message("limbo copy regression sentinel")
        await pilot.pause()
        message = chat.messages[-1]
        await _wait_laid_out(pilot, message)
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(20, 0))
        await pilot.mouse_up(message, offset=(20, 0))
        await _drain_clipboard_worker(pilot)
    assert written, "selection end must push the selected text to the clipboard"
    assert "limbo copy regression" in written[-1]


@pytest.mark.asyncio
async def test_ctrl_c_copies_selection(monkeypatch, tmp_path):
    """The Textual screen binding ctrl+c -> copy_text must reach native tools."""
    written: list[str] = []
    monkeypatch.setattr("limbo.ui.app.write_clipboard_text", _native_stub(written))
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        chat = app.screen.query_one(ChatWidget)
        chat.add_assistant_message("limbo ctrl-c sentinel")
        await pilot.pause()
        message = chat.messages[-1]
        await _wait_laid_out(pilot, message)
        await pilot.mouse_down(message, offset=(0, 0))
        await pilot.hover(message, offset=(15, 0))
        await pilot.mouse_up(message, offset=(15, 0))
        await _drain_clipboard_worker(pilot)
        written.clear()  # ignore the auto-copy from the drag itself
        await pilot.press("ctrl+c")
        await _drain_clipboard_worker(pilot)
    assert written and "limbo ctrl-c" in written[-1]


@pytest.mark.asyncio
async def test_plain_click_does_not_clobber_clipboard(monkeypatch, tmp_path):
    """A click without a drag must not copy (selection is empty)."""
    written: list[str] = []
    monkeypatch.setattr("limbo.ui.app.write_clipboard_text", _native_stub(written))
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.click(offset=(10, 10))
        await _drain_clipboard_worker(pilot)
    assert written == []
