"""User input widget."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from rich.cells import cell_len
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea
from textual.widgets._text_area import EditResult

from limbo.models import Attachment
from limbo.ui import clipboard

# Widget height is text rows + 2 rows of round border. The input starts at
# one text row (height 3) and grows up to MAX_TEXT_ROWS before scrolling.
MIN_TEXT_ROWS = 1
MAX_TEXT_ROWS = 8

# Large pastes (above either threshold, pi-style) are collapsed into a
# one-line placeholder; the real content lives in InputWidget._pastes and
# is expanded back at submit time.
PASTE_COLLAPSE_LINES = 10
PASTE_COLLAPSE_CHARS = 1000

# Matches paste placeholders like ``[粘贴的文本 #1，共 123 行]`` or
# ``[粘贴的文本 #1，1234 字符]``. Anchored to the exact formats the widget
# inserts (no newlines, no free-form middle) so similar-looking pasted
# content is not misdetected.
PASTE_MARKER_RE = re.compile(r"\[粘贴的文本 #(\d+)(?:，共 \d+ 行|，\d+ 字符)\]")


class UserSubmitted(Message):
    """Event emitted when the user submits a message.

    ``force_text`` is the escape hatch for '/'-leading plain text: set when
    the raw input had a leading space before the '/', telling the screen to
    skip slash-command dispatch.
    """

    def __init__(
        self,
        message: str,
        attachments: list[Attachment] | None = None,
        *,
        force_text: bool = False,
    ):
        self.message = message
        self.attachments = attachments or []
        self.force_text = force_text
        super().__init__()


class PasteMarkersInvalid(Message):
    """Posted on submit when placeholder markers have no stored content.

    This happens when a paste placeholder was deleted (removing its stored
    content) and then restored via undo, or when the user hand-types text
    that exactly matches the placeholder format. The marker is submitted
    as literal text; the screen surfaces a warning so the loss is visible.
    """

    def __init__(self, paste_ids: list[int]):
        self.paste_ids = paste_ids
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
        # Shadow TextArea's delete_left/delete_right so paste placeholders
        # are removed atomically (plain text deletions fall through).
        Binding("backspace", "paste_aware_delete_left", show=False, priority=True),
        Binding("delete", "paste_aware_delete_right", show=False, priority=True),
        # Shadows TextArea's built-in ctrl+v (App-clipboard text paste); the
        # action falls back to it explicitly when the OS clipboard holds no
        # image/files.
        Binding("ctrl+v", "paste_attachment", "Paste attachment", show=False),
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
        # Collapsed-paste storage: paste id -> original pasted text. Only
        # ids present here are expanded at submit time (cf. pi's
        # validPasteIds), so lookalike text is never expanded by accident.
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0
        # Pending attachments (clipboard images/files). Each has a marker in
        # the document (``[图片 #N]`` / ``[文件 #N: name]``); at submit time
        # only attachments whose marker is still present are sent.
        self._attachments: list[tuple[str, Attachment]] = []
        self._attachment_counter = 0

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

    # -- paste collapse ----------------------------------------------------------

    @staticmethod
    def _clean_pasted_text(text: str) -> str:
        """Normalize line endings and strip control chars (keep \\n and \\t)."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return "".join(c for c in text if c in "\n\t" or ord(c) >= 32)

    def _insert_replacing_selection(self, text: str) -> None:
        """Insert at the cursor, replacing any active selection.

        Matches Textual's native paste semantics (``_replace_via_keyboard``):
        pasting with an active selection replaces it, at every insertion
        point (raw text, paste markers, attachment markers alike).
        """
        start, end = self.selection
        if start == end:
            self.insert(text)
        else:
            self.replace(text, start, end, maintain_selection_offset=False)

    def on_paste(self, event: events.Paste) -> None:
        """Collapse large pastes into a one-line placeholder (pi-style).

        ``TextArea._on_paste`` (private) would insert the raw text after
        this handler via MRO dispatch; ``prevent_default()`` stops that so
        the widget fully controls what gets inserted.
        """
        event.prevent_default()
        text = self._clean_pasted_text(event.text)
        lines = text.split("\n")
        if len(lines) > PASTE_COLLAPSE_LINES or len(text) > PASTE_COLLAPSE_CHARS:
            self._paste_counter += 1
            paste_id = self._paste_counter
            self._pastes[paste_id] = text
            if len(lines) > PASTE_COLLAPSE_LINES:
                marker = f"[粘贴的文本 #{paste_id}，共 {len(lines)} 行]"
            else:
                marker = f"[粘贴的文本 #{paste_id}，{len(text)} 字符]"
            self._insert_replacing_selection(marker)
            return
        self._insert_replacing_selection(text)

    def _expand_paste_markers(self, text: str) -> tuple[str, list[int]]:
        """Replace placeholders with their stored content.

        Returns the expanded text plus the ids of markers whose content is
        no longer stored (deleted then restored via undo, or hand-typed
        lookalikes); those stay as literal text and are reported so the
        screen can warn the user instead of silently losing the paste.
        """
        invalid: list[int] = []

        def repl(match: re.Match[str]) -> str:
            paste_id = int(match.group(1))
            if paste_id in self._pastes:
                return self._pastes[paste_id]
            invalid.append(paste_id)
            return match.group(0)

        return PASTE_MARKER_RE.sub(repl, text), invalid

    def _valid_marker_ending_at(self, line: str, col: int) -> re.Match[str] | None:
        """A stored-content marker whose text ends exactly at ``col``."""
        for match in PASTE_MARKER_RE.finditer(line):
            if match.end() == col and int(match.group(1)) in self._pastes:
                return match
        return None

    def action_paste_aware_delete_left(self) -> None:
        """Backspace: delete an adjacent paste placeholder atomically."""
        if self.selection[0] == self.selection[1]:
            row, col = self.cursor_location
            match = self._valid_marker_ending_at(self.document.get_line(row), col)
            if match is not None:
                self.delete((row, match.start()), (row, match.end()))
                del self._pastes[int(match.group(1))]
                return
        self.action_delete_left()

    def action_paste_aware_delete_right(self) -> None:
        """Delete: remove an adjacent paste placeholder atomically."""
        if self.selection[0] == self.selection[1]:
            row, col = self.cursor_location
            match = PASTE_MARKER_RE.match(self.document.get_line(row), col)
            if match is not None and int(match.group(1)) in self._pastes:
                self.delete((row, match.start()), (row, match.end()))
                del self._pastes[int(match.group(1))]
                return
        self.action_delete_right()

    def clear(self) -> EditResult:
        result = super().clear()
        # Paste/attachment state is per-message; reset it on the same
        # clear() path (regular and slash-command submits alike) so stale
        # ids can't leak into the next message.
        self._pastes.clear()
        self._attachments.clear()
        return result

    # -- attachments -------------------------------------------------------------

    def action_paste_attachment(self) -> None:
        """Ctrl+V: attach clipboard images/files, else defer to text paste.

        The clipboard probe shells out to platform tools (on macOS several
        osascript calls, each with its own timeout), so it runs off the
        event loop via ``asyncio.to_thread``; results are applied back on
        the main thread.

        Degradation chain (RFC v2 §4.3.1): image/files → attachment markers;
        text/empty/unreadable → TextArea's original ``action_paste`` (the
        App-clipboard text path), never a silent no-op.
        """
        self.run_worker(self._paste_os_clipboard())

    async def _paste_os_clipboard(self) -> None:
        content = await asyncio.to_thread(clipboard.read_clipboard)
        if isinstance(content, clipboard.ClipboardImage):
            path = clipboard.save_clipboard_image(content.data, content.ext)
            self._attachment_counter += 1
            n = self._attachment_counter
            attachment = Attachment(
                kind="image",
                name=f"剪贴板图片-{n}.{content.ext}",
                path=str(path),
                mime=f"image/{content.ext}",
            )
            marker = f"[图片 #{n}]"
            self._attachments.append((marker, attachment))
            self._insert_replacing_selection(marker)
            return
        if isinstance(content, clipboard.ClipboardFiles):
            for path in content.paths:
                self._attachment_counter += 1
                n = self._attachment_counter
                attachment = Attachment(
                    kind="file", name=path.name, path=str(path)
                )
                marker = f"[文件 #{n}: {path.name}]"
                self._attachments.append((marker, attachment))
                self._insert_replacing_selection(marker)
            return
        # Text/empty/failure: keep the built-in text paste behavior.
        super().action_paste()

    # -- submission ------------------------------------------------------------

    def action_submit(self) -> None:
        screen = self._menu_screen()
        if screen is not None and screen.slash_menu_complete(execute=True):
            return
        raw = self.text
        # Escape hatch: a leading space before a '/' forces plain text, so
        # the message skips slash-command dispatch (the stripped text is
        # what gets sent; the screen's unknown-command error hints at this).
        force_text = raw[:1].isspace() and raw.lstrip().startswith("/")
        text, invalid_ids = self._expand_paste_markers(raw.strip())
        if invalid_ids:
            self.post_message(PasteMarkersInvalid(invalid_ids))
        if text:
            # Only send attachments whose marker survived editing.
            attachments = [a for marker, a in self._attachments if marker in text]
            if not self._history or self._history[-1] != text:
                self._history.append(text)
            self._history_index = None
            self._history_value = None
            self._draft = ""
            self.post_message(UserSubmitted(text, attachments, force_text=force_text))
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
            return
        # Menu closed: Esc routes through the screen's escape chain
        # (verify → interrupt turn → cancel newest queued steer; RFC
        # LIM-20/LIM-40/LIM-53). The dispatch lives here — not in a
        # screen-level binding — because this priority binding always sees
        # Esc first while the input is focused.
        handle = getattr(self.screen, "handle_escape", None)
        if handle is not None:
            handle()
