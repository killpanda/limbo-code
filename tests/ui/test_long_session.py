"""Long-session lag guards: frozen replies, reflow-free line updates, GC."""

from __future__ import annotations

import gc

import pytest
from textual.app import App
from textual.widgets import Markdown, Static

from limbo.ui import heap
from limbo.ui.reflow import update_line
from limbo.ui.theme import LIMBO_DUSK
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.markdown_view import MarkdownView, render_markdown, stable_end

REPLY = "## Plan\n\n- read `a.py`\n- fix it\n\n```python\nx = 1\n```\n\nDone."


class ChatApp(App[None]):
    def compose(self):
        yield ChatWidget(id="chat")


async def wait_until(pilot, predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("condition not reached within timeout")


@pytest.mark.asyncio
async def test_finished_reply_is_frozen_into_one_widget():
    app = ChatApp()
    async with app.run_test(size=(80, 30)) as pilot:
        chat = app.query_one(ChatWidget)
        await chat.append_assistant_text(REPLY)
        await chat.flush_stream()
        await pilot.pause()
        (view,) = chat.query(MarkdownView)
        assert view.streaming
        assert not chat.query(Markdown)
        streamed_height = view.region.height

        await chat.end_stream()
        await pilot.pause()
        assert not view.streaming
        assert chat.messages == [view]
        assert view.source == REPLY
        assert "assistant-message" in view.classes
        assert view.region.height == streamed_height
        assert chat.transcript_text() == REPLY


@pytest.mark.asyncio
async def test_buffered_tail_stays_with_its_reply():
    """Text still buffered when a tool card starts belongs to the old reply."""
    app = ChatApp()
    async with app.run_test(size=(80, 30)):
        chat = app.query_one(ChatWidget)
        await chat.append_assistant_text("first reply")
        chat.add_tool_card("t1", "read", {"path": "a.py"})
        await chat.append_assistant_text("second reply")
        await chat.end_stream()

        replies = [m.source for m in chat.messages if isinstance(m, MarkdownView)]
        assert replies == ["first reply", "second reply"]


@pytest.mark.asyncio
async def test_restored_reply_mounts_frozen():
    app = ChatApp()
    async with app.run_test(size=(80, 30)) as pilot:
        chat = app.query_one(ChatWidget)
        chat.add_assistant_message(REPLY)
        await pilot.pause()
        assert not chat.query(Markdown)
        frozen = chat.query_one(MarkdownView)
        assert frozen.region.height > 0


@pytest.mark.asyncio
async def test_frozen_reply_rewraps_on_resize():
    app = ChatApp()
    async with app.run_test(size=(80, 30)) as pilot:
        chat = app.query_one(ChatWidget)
        chat.add_assistant_message("word " * 60)
        await pilot.pause()
        frozen = chat.query_one(MarkdownView)
        wide = frozen.region.height
        await pilot.resize_terminal(40, 30)
        await wait_until(pilot, lambda: frozen.region.height > wide)


def test_render_markdown_matches_theme_tokens():
    variables = {
        "foreground": LIMBO_DUSK.foreground,
        "primary": LIMBO_DUSK.primary,
        "accent": LIMBO_DUSK.accent,
        "panel": LIMBO_DUSK.panel,
        **LIMBO_DUSK.variables,
    }
    text = render_markdown(REPLY, 40, LIMBO_DUSK, variables)
    lines = text.plain.split("\n")
    assert lines[0].rstrip() == "Plan"  # left-aligned heading
    assert any(line.startswith("▌ x = 1") for line in lines)  # fence bar
    assert not lines[-1].strip() == ""  # no trailing blank separator
    colors = {
        span.style.color.triplet.hex
        for span in text.spans
        if not isinstance(span.style, str) and span.style.color is not None
    }
    assert str(LIMBO_DUSK.accent).lower() in colors
    # A plain name in code is themed, not Pygments' #000000 fallback.
    assert "#000000" not in colors


@pytest.mark.asyncio
async def test_update_line_skips_reflow_when_size_is_stable():
    class LineApp(App[None]):
        def compose(self):
            yield Static("⠋ thinking 1s", id="line")

    app = LineApp()
    async with app.run_test(size=(60, 5)) as pilot:
        line = app.query_one("#line", Static)
        update_line(line, "⠋ thinking 1s")  # first call: unknown footprint
        await pilot.pause()
        update_line(line, "⠙ thinking 2s")
        assert not line._layout_required
        update_line(line, "⠙ thinking 10s")  # still fits the full-width line
        assert not line._layout_required
        update_line(line, "x" * 80)  # wraps: the height changes
        assert line._layout_required


def test_heap_keeper_freezes_and_sweeps_when_quiet(monkeypatch):
    monkeypatch.setattr(heap, "QUIET_OBJECTS", 10**9)  # every tick is quiet
    monkeypatch.setattr(heap, "QUIET_TICKS", 2)
    try:
        keeper = heap.HeapKeeper()
        keeper._dirty = True
        keeper.tick()
        assert gc.get_freeze_count() > 0
        frozen = gc.get_freeze_count()
        keeper.tick()  # second quiet tick: one full sweep, then re-freeze
        assert keeper._dirty is False
        assert gc.get_freeze_count() <= frozen + 1000
    finally:
        gc.unfreeze()


def test_stable_end_keeps_open_fences_in_the_tail():
    src = "para one\n\n```python\nx = 1\n\ny = 2\n"
    assert src[: stable_end(src)] == "para one\n\n"
    done = src + "```\n\nnext"
    assert done[stable_end(done) :] == "next"


@pytest.mark.asyncio
async def test_streamed_blocks_render_once():
    app = ChatApp()
    async with app.run_test(size=(80, 30)) as pilot:
        chat = app.query_one(ChatWidget)
        await chat.append_assistant_text("first block\n\n")
        await chat.flush_stream()
        await pilot.pause()
        view = chat.query_one(MarkdownView)
        first = list(view._chunks)
        await chat.append_assistant_text("second ")
        await chat.flush_stream()
        await pilot.pause()
        await chat.append_assistant_text("block")
        await chat.flush_stream()
        await pilot.pause()
        assert view._chunks == first  # completed block not re-rendered
        assert "second block" in str(view.render())
