"""User input widget."""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea

# Widget height is text rows + 2 rows of round border. The input starts at
# one text row (height 3) and grows up to MAX_TEXT_ROWS before scrolling.
MIN_TEXT_ROWS = 1
MAX_TEXT_ROWS = 8


class UserSubmitted(Message):
    """Event emitted when the user submits a message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__()


class InputWidget(TextArea):
    """Multi-line input that submits on Enter (Shift+Enter for newline).

    The widget grows vertically as lines are added (up to MAX_TEXT_ROWS),
    and recalls previously submitted input with the up/down arrow keys
    (shell-style history, in-memory per session).

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
        # Shell-style input history (in-memory, per session).
        self._history: list[str] = []
        self._history_index: int | None = None
        self._draft: str = ""
        # Last value applied via history recall; used to tell recall edits
        # apart from manual edits in the (async) Changed handler.
        self._history_value: str | None = None

    def _menu_screen(self):
        """The screen, but only when its slash-command menu is open."""
        screen = self.screen
        if getattr(screen, "slash_menu_open", False):
            return screen
        return None

    # -- auto-growing height ---------------------------------------------------

    def _visual_rows(self) -> int:
        """Number of display rows the current text occupies (soft-wrapped)."""
        width = self.wrap_width
        rows = 0
        for line in self.document.lines:
            length = cell_len(line)
            if width > 0:
                rows += max(1, -(-length // width))
            else:
                rows += 1
        return max(MIN_TEXT_ROWS, rows)

    def _auto_resize(self) -> None:
        height = min(self._visual_rows(), MAX_TEXT_ROWS) + 2
        current = self.styles.height
        if current is not None and getattr(current, "value", None) == height:
            return
        self.styles.height = height

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area is not self:
            return
        self._auto_resize()
        # Editing by hand while browsing history cancels the recall position,
        # so the next up/down starts fresh from the latest entry. Changes
        # applied by history recall itself keep the navigation position.
        if self._history_value is not None and self.text == self._history_value:
            return
        self._history_value = None
        self._history_index = None

    def on_resize(self) -> None:
        # Terminal width changes affect soft-wrapping, hence the row count.
        self._auto_resize()

    # -- submission ------------------------------------------------------------

    def action_submit(self) -> None:
        screen = self._menu_screen()
        if screen is not None and screen.slash_menu_complete(execute=True):
            return
        text = self.text.strip()
        if text:
            if not self._history or self._history[-1] != text:
                self._history.append(text)
            self._history_index = None
            self._history_value = None
            self._draft = ""
            self.post_message(UserSubmitted(text))
            self.clear()

    def action_newline(self) -> None:
        """Insert a newline at the current cursor position."""
        self.insert("\n")

    # -- history recall --------------------------------------------------------

    def _set_text_from_history(self, value: str) -> None:
        self._history_value = value
        self.text = value
        self.move_cursor(self.document.end)

    def _cursor_on_first_visual_row(self) -> bool:
        """Whether the cursor sits on the first *visual* (soft-wrapped) row.

        With soft wrap a single logical line spans several display rows; the
        logical row from ``cursor_location`` stays 0, so the column is used
        to approximate the visual position (wide chars make this approximate).
        """
        row, col = self.cursor_location
        if row > 0:
            return False
        width = self.wrap_width
        return width <= 0 or col < width

    def _history_prev(self) -> None:
        if not self._history:
            self.action_cursor_up()
            return
        if self._history_index is None:
            # Multi-line/soft-wrapped input: up moves the cursor until it
            # reaches the first visual row; only then does it start
            # recalling history.
            if not self._cursor_on_first_visual_row():
                self.action_cursor_up()
                return
            self._draft = self.text
            self._history_index = len(self._history) - 1
        else:
            self._history_index = max(0, self._history_index - 1)
        self._set_text_from_history(self._history[self._history_index])

    def _history_next(self) -> None:
        if self._history_index is None:
            self.action_cursor_down()
            return
        self._history_index += 1
        if self._history_index >= len(self._history):
            self._history_index = None
            self._set_text_from_history(self._draft)
        else:
            self._set_text_from_history(self._history[self._history_index])

    # -- slash-menu key redirection --------------------------------------------

    def action_menu_up(self) -> None:
        screen = self._menu_screen()
        if screen is None:
            self._history_prev()
        else:
            screen.slash_menu_move(-1)

    def action_menu_down(self) -> None:
        screen = self._menu_screen()
        if screen is None:
            self._history_next()
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
