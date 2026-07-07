"""Chat message widget."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class ChatWidget(VerticalScroll):
    """Displays conversation messages."""

    DEFAULT_CSS = """
    ChatWidget {
        width: 1fr;
        height: 1fr;
    }
    .user-message {
        color: $text-accent;
        text-align: right;
    }
    .assistant-message {
        color: $text;
        text-align: left;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages: list[Static] = []

    def add_user_message(self, text: str) -> None:
        msg = Static(f"You: {text}", classes="user-message")
        self.messages.append(msg)
        self.mount(msg)

    def add_assistant_text(self, text: str) -> None:
        msg = Static(text, classes="assistant-message", markup=False)
        self.messages.append(msg)
        self.mount(msg)

    def append_assistant_text(self, text: str) -> None:
        if self.messages and "assistant-message" in self.messages[-1].classes:
            current = self.messages[-1]
            current.update(str(current.content) + text)
        else:
            self.add_assistant_text(text)
