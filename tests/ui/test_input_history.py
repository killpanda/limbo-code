"""Tests for InputWidget auto-grow and in-session input history."""

import pytest
from textual.app import App

from limbo.ui.widgets.input import MAX_TEXT_ROWS, InputWidget, UserSubmitted


def make_app(submitted: list | None = None):
    class TestApp(App[None]):
        def compose(self):
            yield InputWidget(id="input")

        def on_user_submitted(self, event: UserSubmitted) -> None:
            if submitted is not None:
                submitted.append(event)

    return TestApp()


async def submit(pilot, widget, text: str) -> None:
    widget.text = text
    widget.action_submit()
    await pilot.pause()


@pytest.mark.asyncio
async def test_up_recalls_last_submission():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "first message")

        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "first message"


@pytest.mark.asyncio
async def test_history_navigation_and_draft_restore():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "one")
        await submit(pilot, widget, "two")

        # Type a draft, then walk back through history.
        widget.text = "draft in progress"
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "two"
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "one"
        # Up at the oldest entry stays put.
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "one"
        # Down walks forward, then restores the draft.
        await pilot.press("down")
        await pilot.pause()
        assert widget.text == "two"
        await pilot.press("down")
        await pilot.pause()
        assert widget.text == "draft in progress"


@pytest.mark.asyncio
async def test_consecutive_duplicates_stored_once():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "same")
        await submit(pilot, widget, "same")
        await submit(pilot, widget, "other")

        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "other"
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "same"
        # No duplicate "same" entry above this one.
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "same"


@pytest.mark.asyncio
async def test_up_on_second_row_moves_cursor_not_history():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "earlier")

        widget.text = "line one\nline two"
        widget.move_cursor((1, 0))
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "line one\nline two"
        assert widget.cursor_location == (0, 0)
        # Now on the first row: up recalls history.
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "earlier"


@pytest.mark.asyncio
async def test_manual_edit_cancels_history_navigation():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "one")
        await submit(pilot, widget, "two")

        await pilot.press("up")  # recalls "two"
        await pilot.pause()
        widget.insert(" edited")
        await pilot.pause()
        # Down should behave as plain cursor movement now, not clobber the
        # edit by walking forward in history.
        await pilot.press("down")
        await pilot.pause()
        assert widget.text == "two edited"


@pytest.mark.asyncio
async def test_height_grows_with_lines_and_shrinks_on_clear():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.text = "a\nb\nc"
        await pilot.pause()
        assert widget.styles.height.value == 5  # 3 rows + 2 border rows

        widget.clear()
        await pilot.pause()
        assert widget.styles.height.value == 3


@pytest.mark.asyncio
async def test_height_capped_at_max_rows():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.text = "\n".join(f"line {i}" for i in range(MAX_TEXT_ROWS + 5))
        await pilot.pause()
        assert widget.styles.height.value == MAX_TEXT_ROWS + 2


@pytest.mark.asyncio
async def test_height_grows_on_soft_wrap():
    app = make_app()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        wrap_width = widget.wrap_width
        assert wrap_width > 0
        widget.text = "x" * (wrap_width * 2 + 1)  # wraps to 3 visual rows
        await pilot.pause()
        assert widget.styles.height.value == 5


@pytest.mark.asyncio
async def test_up_on_soft_wrapped_line_moves_cursor_visually_first():
    """On a soft-wrapped single line, up walks visual rows before recalling."""
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "earlier")

        wrap_width = widget.wrap_width
        long_line = "x" * (wrap_width * 2 + 1)  # wraps to 3 visual rows
        widget.text = long_line
        widget.move_cursor(widget.document.end)
        await pilot.pause()
        assert widget.cursor_location[1] > wrap_width  # visual row 2

        # First up: cursor moves up a visual row, no history recall.
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == long_line
        assert widget.cursor_location[1] < wrap_width * 2

        # Walk to the first visual row, then up recalls history.
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == long_line
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "earlier"


@pytest.mark.asyncio
async def test_submit_resets_history_navigation():
    submitted = []
    app = make_app(submitted)
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        await submit(pilot, widget, "first")
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "first"
        # Resubmit the recalled entry: up afterwards recalls it again.
        await pilot.press("enter")
        await pilot.pause()
        assert widget.text == ""
        await pilot.press("up")
        await pilot.pause()
        assert widget.text == "first"
