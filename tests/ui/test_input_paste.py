"""Tests for InputWidget large-paste collapse (pi-style placeholders)."""

import pytest
from textual import events
from textual.app import App

from limbo.ui.widgets.input import (
    PASTE_COLLAPSE_CHARS,
    PASTE_COLLAPSE_LINES,
    InputWidget,
    PasteMarkersInvalid,
    UserSubmitted,
)


def make_app(submitted: list | None = None, invalid: list | None = None):
    class TestApp(App[None]):
        def compose(self):
            yield InputWidget(id="input")

        def on_user_submitted(self, event: UserSubmitted) -> None:
            if submitted is not None:
                submitted.append(event.message)

        def on_paste_markers_invalid(self, event: PasteMarkersInvalid) -> None:
            if invalid is not None:
                invalid.append(event.paste_ids)

    return TestApp()


async def paste(pilot, widget, text: str) -> None:
    """Simulate a bracketed paste (the App forwards it to the focused widget)."""
    pilot.app.post_message(events.Paste(text))
    await pilot.pause()


async def submit(pilot, widget) -> None:
    widget.action_submit()
    await pilot.pause()


def big_lines(n: int = PASTE_COLLAPSE_LINES + 1) -> str:
    return "\n".join(f"log line {i} with some content" for i in range(n))


def big_chars(n: int = PASTE_COLLAPSE_CHARS + 1) -> str:
    return "x" * n


@pytest.mark.asyncio
async def test_small_paste_inserted_verbatim():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await paste(pilot, widget, "hello world")
        assert widget.text == "hello world"
        assert widget._pastes == {}

        small_multiline = "line1\nline2\nline3"
        await paste(pilot, widget, small_multiline)
        assert small_multiline in widget.text
        assert widget._pastes == {}


@pytest.mark.asyncio
async def test_large_paste_collapses_to_placeholder_only():
    """Regression: interception must prevent TextArea's default insertion,
    so the document holds only the placeholder, never the original text."""
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        original = big_lines()
        await paste(pilot, widget, original)
        assert widget.text == f"[粘贴的文本 #1，共 {PASTE_COLLAPSE_LINES + 1} 行]"
        assert "log line" not in widget.text
        assert widget._pastes == {1: original}


@pytest.mark.asyncio
async def test_long_single_line_paste_collapses_by_chars():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        original = big_chars()
        await paste(pilot, widget, original)
        assert widget.text == f"[粘贴的文本 #1，{PASTE_COLLAPSE_CHARS + 1} 字符]"
        assert widget._pastes == {1: original}


@pytest.mark.asyncio
async def test_paste_cleans_control_chars_and_crlf():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await paste(pilot, widget, "a\r\nb\x00c\x07")
        assert widget.text == "a\nbc"


@pytest.mark.asyncio
async def test_submit_expands_placeholder():
    submitted = []
    invalid = []
    app = make_app(submitted, invalid)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        original = big_lines()
        await paste(pilot, widget, original)
        await submit(pilot, widget)
        assert submitted == [original]
        assert invalid == []
        # clear() after submit resets paste state.
        assert widget._pastes == {}
        assert widget.text == ""


@pytest.mark.asyncio
async def test_multiple_pastes_numbered_and_expanded_independently():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        first, second = big_lines(), big_chars()
        await paste(pilot, widget, first)
        widget.insert(" middle ")
        await paste(pilot, widget, second)
        assert "[粘贴的文本 #1，共 11 行]" in widget.text
        assert "[粘贴的文本 #2，1001 字符]" in widget.text
        await submit(pilot, widget)
        assert submitted == [f"{first} middle {second}"]


@pytest.mark.asyncio
async def test_lookalike_marker_semantics_locked():
    """Locked semantics (RFC v2 §4.2.5): text that exactly matches the
    placeholder format but has no stored content is sent literally AND
    reported as invalid — a conservative behavior on purpose."""
    submitted = []
    invalid = []
    app = make_app(submitted, invalid)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "看看这个 [粘贴的文本 #9，共 20 行] 谢谢"
        await submit(pilot, widget)
        assert submitted == ["看看这个 [粘贴的文本 #9，共 20 行] 谢谢"]
        assert invalid == [[9]]


@pytest.mark.asyncio
async def test_near_miss_marker_format_not_flagged():
    """Similar but non-exact formats are neither expanded nor flagged."""
    submitted = []
    invalid = []
    app = make_app(submitted, invalid)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "[粘贴的文本 #9] 和 [粘贴的文本 #9，共 abc 行]"
        await submit(pilot, widget)
        assert submitted == ["[粘贴的文本 #9] 和 [粘贴的文本 #9，共 abc 行]"]
        assert invalid == []


@pytest.mark.asyncio
async def test_backspace_deletes_placeholder_atomically():
    submitted = []
    invalid = []
    app = make_app(submitted, invalid)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        original = big_lines()
        await paste(pilot, widget, original)
        # Cursor is right after the inserted placeholder.
        await pilot.press("backspace")
        await pilot.pause()
        assert widget.text == ""
        assert widget._pastes == {}
        # Submitting nothing posts no warning and no message.
        await submit(pilot, widget)
        assert submitted == []
        assert invalid == []
        assert original  # silence unused warning


@pytest.mark.asyncio
async def test_backspace_away_from_placeholder_behaves_normally():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "abc"
        widget.move_cursor(widget.document.end)
        await pilot.press("backspace")
        await pilot.pause()
        assert widget.text == "ab"


@pytest.mark.asyncio
async def test_delete_key_removes_placeholder_to_the_right():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await paste(pilot, widget, big_lines())
        widget.move_cursor((0, 0))
        await pilot.press("delete")
        await pilot.pause()
        assert widget.text == ""
        assert widget._pastes == {}


@pytest.mark.asyncio
async def test_undo_restored_placeholder_warns_on_submit():
    """Atomic delete -> ctrl+z restores the placeholder text, but its
    stored content is gone: submit warns and sends the marker literally."""
    submitted = []
    invalid = []
    app = make_app(submitted, invalid)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await paste(pilot, widget, big_lines())
        await pilot.press("backspace")  # atomic delete
        await pilot.pause()
        assert widget.text == ""
        await pilot.press("ctrl+z")  # undo restores the placeholder text
        await pilot.pause()
        assert widget.text == "[粘贴的文本 #1，共 11 行]"
        assert widget._pastes == {}
        await submit(pilot, widget)
        assert submitted == ["[粘贴的文本 #1，共 11 行]"]
        assert invalid == [[1]]


@pytest.mark.asyncio
async def test_undo_after_paste_does_not_expand_stale_id():
    """Paste -> undo removes the placeholder; the stale _pastes entry must
    not expand anything on a later submit."""
    submitted = []
    invalid = []
    app = make_app(submitted, invalid)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await paste(pilot, widget, big_lines())
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert widget.text == ""
        widget.text = "plain message"
        await submit(pilot, widget)
        assert submitted == ["plain message"]
        assert invalid == []


@pytest.mark.asyncio
async def test_history_recall_returns_expanded_text():
    app = make_app([])
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        original = big_lines()
        await paste(pilot, widget, original)
        await submit(pilot, widget)
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == original


@pytest.mark.asyncio
async def test_placeholder_does_not_grow_input_height():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await paste(pilot, widget, big_lines(200))
        await pilot.pause()
        # A single placeholder line keeps the widget at minimum height.
        assert widget._visual_rows() == 1


@pytest.mark.asyncio
async def test_small_paste_replaces_selection():
    """Native Textual semantics: pasting over a selection replaces it."""
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "hello world"
        widget.selection = ((0, 6), (0, 11))
        await paste(pilot, widget, "SMALL")
        assert widget.text == "hello SMALL"


@pytest.mark.asyncio
async def test_large_paste_marker_replaces_selection():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "hello world"
        widget.selection = ((0, 6), (0, 11))
        original = big_lines()
        await paste(pilot, widget, original)
        assert widget.text == "hello [粘贴的文本 #1，共 11 行]"
        assert "world" not in widget.text
        assert widget._pastes == {1: original}
