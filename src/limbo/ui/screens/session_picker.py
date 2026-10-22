"""Session picker modal: list sessions and let the user switch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Label, ListItem, ListView

from limbo.sessions import SessionMeta
from limbo.ui.screens.dialog import Dialog
from limbo.ui.widgets.landing import relative_time


class SessionPicker(Dialog[Path | None]):
    """Modal list of sessions; Enter switches, Esc cancels.

    Dismisses with the selected session path, or None when cancelled.
    """

    TITLE_TEXT = "会话"
    HINTS = (("↑↓", "选择"), ("enter", "切换"), ("esc", "取消"))

    def __init__(self, sessions: list[SessionMeta], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sessions = sessions

    def compose_body(self) -> ComposeResult:
        yield ListView(
            *[ListItem(Label(self._format(meta), markup=False)) for meta in self._sessions]
        )

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one(ListView).focus()

    def _format(self, meta: SessionMeta) -> Text:
        v = self.app.current_theme.variables
        title = meta.title or "（无标题）"
        text = Text(title, Style(bold=True))
        # list_sessions surfaces the file mtime as updated_at, so this shows
        # when the session was last written even under incremental saves.
        when = relative_time(meta.updated_at) or meta.updated_at[:16].replace("T", " ")
        text.append(f"  {when}", Style(color=v.get("text-secondary", "")))
        text.append(f"  {meta.id}", Style(color=v.get("text-tertiary", "")))
        if meta.parent_session:
            # /fork or /handoff child: show where it branched from.
            text.append(
                f"  ⑂ {meta.parent_session}", Style(color=v.get("text-tertiary", ""))
            )
        text.no_wrap = True
        text.overflow = "ellipsis"
        return text

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is not None and 0 <= index < len(self._sessions):
            self.dismiss(self._sessions[index].path)
