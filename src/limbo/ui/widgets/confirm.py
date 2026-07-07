"""Confirmation modal for file writes."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class Confirmed(Message):
    """User approved the action."""


class Rejected(Message):
    """User rejected the action."""


class ConfirmDialog(ModalScreen[None]):
    """Modal dialog showing a diff and asking for approval."""

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }
    ConfirmDialog > Vertical {
        width: 80;
        height: auto;
        max-height: 80vh;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }
    ConfirmDialog #diff-body {
        max-height: 50vh;
    }
    """

    def __init__(self, title: str, body: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title_text = title
        self.body_text = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text)
            with VerticalScroll(id="diff-body"):
                yield Label(self.body_text)
            with Horizontal():
                yield Button("Apply", variant="success", id="apply")
                yield Button("Reject", variant="error", id="reject")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.post_message(Confirmed())
        else:
            self.post_message(Rejected())
        self.dismiss()
