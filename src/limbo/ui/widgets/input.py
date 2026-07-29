"""User input widget."""

from __future__ import annotations

from typing import Any

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class UserSubmitted(Message):
    """Event emitted when the user submits a message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__()


class InputWidget(TextArea):
    """Multi-line input that submits on Enter (Shift+Enter for newline).

    When the screen's slash-command menu is open, Enter/Tab/up/down/Esc are
    redirected to the menu instead of their default behavior.

    Styles live in ``limbo/ui/app.tcss``.
    """

    # Use priority bindings so these actions run before TextArea's default
    # key handling, which would otherwise insert a newline on Enter.
    BINDINGS = [
        Binding("enter", "submit", "Submit", priority=True),
        Binding("shift+enter", "newline", "Newline", priority=True),
        Binding("up", "menu_up", show=False, priority=True),
        Binding("down", "menu_down", show=False, priority=True),
        Binding("tab", "menu_complete", show=False, priority=True),
        Binding("escape", "menu_cancel", show=False, priority=True),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.show_line_numbers = False

    def _menu_screen(self):
        """The screen, but only when its slash-command menu is open."""
        screen = self.screen
        if getattr(screen, "slash_menu_open", False):
            return screen
        return None

    def action_submit(self) -> None:
        screen = self._menu_screen()
        if screen is not None and screen.slash_menu_complete(execute=True):
            return
        text = self.text.strip()
        if text:
            self.post_message(UserSubmitted(text))
            self.clear()

    def action_newline(self) -> None:
        """Insert a newline at the current cursor position."""
        self.insert("\n")

    def action_menu_up(self) -> None:
        screen = self._menu_screen()
        if screen is None:
            self.action_cursor_up()
        else:
            screen.slash_menu_move(-1)

    def action_menu_down(self) -> None:
        screen = self._menu_screen()
        if screen is None:
            self.action_cursor_down()
        else:
            screen.slash_menu_move(1)

    def action_menu_complete(self) -> None:
        screen = self._menu_screen()
        if screen is None:
            self.screen.focus_next()
        else:
            screen.slash_menu_complete(execute=False)

    def action_menu_cancel(self) -> None:
        screen = self._menu_screen()
        if screen is not None:
            screen.slash_menu_close()
