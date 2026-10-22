"""Tests for InputWidget large-paste collapse (pi-style placeholders)."""

import pytest
from textual import events
from textual.app import App

from limbo.ui.widgets.input import (
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
        assert widget._store._pastes == {}
        assert widget.text == ""


