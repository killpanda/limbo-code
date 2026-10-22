"""Limbo Dusk dialog base: a veiled modal with a ``title ╱╱╱`` header.

Every overlay (command palette, sessions, models, auth) shares this frame
so they read as one system: dim veil over the app, a ``$surface`` card,
the title lit by the theme gradient followed by a ``╱`` rule, and a
one-line key hint at the bottom. Esc always cancels.
"""

from __future__ import annotations

from typing import Any, TypeVar

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from limbo.ui.gradient import gradient_text, theme_gradient

T = TypeVar("T")

_DIAG = "╱" * 200


def dialog_title(title: str, theme: Any) -> Text:
    v = theme.variables
    text = gradient_text(title, theme_gradient(theme), bold=True).copy()
    text.append(" ")
    text.append(_DIAG, Style(color=v.get("diag", v.get("text-tertiary", ""))))
    text.no_wrap = True
    text.overflow = "crop"
    return text


def key_hint(pairs: tuple[tuple[str, str], ...], theme: Any) -> Text:
    v = theme.variables
    text = Text()
    for i, (key, desc) in enumerate(pairs):
        if i:
            text.append("  ")
        text.append(key, Style(color=v.get("text-secondary", ""), bold=True))
        text.append(f" {desc}", Style(color=v.get("text-tertiary", "")))
    return text


class Dialog(ModalScreen[T]):
    """Base modal. Subclasses set ``TITLE_TEXT``/``HINTS`` and ``compose_body``."""

    TITLE_TEXT = ""
    HINTS: tuple[tuple[str, str], ...] = (("esc", "关闭"),)
    BINDINGS = [Binding("escape", "cancel", "取消", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(classes="dialog-title", markup=False)
            yield from self.compose_body()
            yield Static(classes="dialog-hint", markup=False)

    def compose_body(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, lambda _t: self._render_frame())
        self._render_frame()

    def _render_frame(self) -> None:
        theme = self.app.current_theme
        self.query_one(".dialog-title", Static).update(dialog_title(self.TITLE_TEXT, theme))
        self.query_one(".dialog-hint", Static).update(key_hint(self.HINTS, theme))

    def action_cancel(self) -> None:
        self.dismiss(None)
