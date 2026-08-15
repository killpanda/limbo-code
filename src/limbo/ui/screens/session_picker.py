"""Session picker modal: list sessions and let the user switch."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

from limbo.sessions import SessionMeta


class SessionPicker(Screen[Path | None]):
    """Modal list of sessions; Enter switches, Esc cancels.

    Dismisses with the selected session path, or None when cancelled.
    """

    BINDINGS = [
        Binding("escape", "cancel", "取消"),
    ]

    def __init__(self, sessions: list[SessionMeta], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker"):
            yield Static(
                "选择会话 · Enter 切换 · Esc 取消",
                id="picker-title",
                markup=False,
            )
            yield ListView(
                *[
                    ListItem(Label(self._format(meta), markup=False))
                    for meta in self._sessions
                ]
            )

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @staticmethod
    def _format(meta: SessionMeta) -> str:
        title = meta.title or "(无标题)"
        # list_sessions surfaces the file mtime as updated_at, so this shows
        # when the session was last written even under incremental saves.
        updated = meta.updated_at[:16].replace("T", " ")
        return f"{title}  ·  {updated}  ·  {meta.id}"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is not None and 0 <= index < len(self._sessions):
            self.dismiss(self._sessions[index].path)

    def action_cancel(self) -> None:
        self.dismiss(None)
