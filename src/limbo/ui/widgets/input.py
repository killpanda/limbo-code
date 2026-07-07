"""User input widget."""

from __future__ import annotations

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_line_numbers = False

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self.post_message(UserSubmitted(text))
            self.clear()
