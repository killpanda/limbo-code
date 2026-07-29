"""Confirmation modal for destructive tool actions."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class Confirmed(Message):
    """User approved the action."""


class Rejected(Message):
    """User rejected the action."""


class ConfirmDialog(ModalScreen[None]):
    """Modal dialog showing a diff/preview and asking for approval."""

    BINDINGS = [
        Binding("y", "apply", "Apply"),
        Binding("n", "reject", "Reject"),
        Binding("escape", "reject", "Reject"),
    ]

    def __init__(
        self,
        title: str,
        body: str,
        warning: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.title_text = title
        self.body_text = body
        self.warning_text = warning

    def action_apply(self) -> None:
        self.post_message(Confirmed())
        self.dismiss()

    def action_reject(self) -> None:
        self.post_message(Rejected())
        self.dismiss()

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text, id="dialog-title", markup=False)
            with VerticalScroll(id="diff-body"):
                yield Label(self.body_text, markup=False)
            if self.warning_text:
                yield Label(
                    f"⚠ {self.warning_text}", id="dialog-warning", markup=False
                )
            with Horizontal(id="dialog-buttons"):
                yield Button("[Y] Apply", variant="success", id="apply")
                yield Button("[N] Reject", variant="error", id="reject")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.post_message(Confirmed())
        else:
            self.post_message(Rejected())
        self.dismiss()
