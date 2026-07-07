"""User input widget."""

from __future__ import annotations

from typing import Any

from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.widgets import TextArea


class UserSubmitted(Message):
    """Event emitted when the user submits a message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__()


class InputWidget(TextArea):
    """Multi-line input that submits on Enter (Shift+Enter for newline)."""

    DEFAULT_CSS = """
    InputWidget {
        height: 3;
    }
    """

    BINDINGS = [
        Binding("enter", "submit", "Submit"),
        Binding("shift+enter", "newline", "Newline"),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.show_line_numbers = False

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self.post_message(UserSubmitted(text))
            self.clear()

    def action_newline(self) -> None:
        """Insert a newline at the current cursor position."""
        self.insert("\n")

    async def _on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.action_submit()
        elif event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.action_newline()
        else:
            await super()._on_key(event)
