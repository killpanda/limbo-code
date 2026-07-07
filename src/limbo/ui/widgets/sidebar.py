"""Sidebar showing session info and recent files."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Static


class SidebarWidget(Vertical):
    """Left sidebar."""

    DEFAULT_CSS = """
    SidebarWidget {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = Static("Limbo", id="session-title")
        self.recent_files = Static("Recent files:\n(none)", id="recent-files")
        self.status = Static("Idle", id="tool-status")

    def compose(self):
        yield self.title
        yield self.recent_files
        yield self.status

    def set_status(self, text: str) -> None:
        self.status.update(text)

    def add_recent_file(self, path: str) -> None:
        current = str(self.recent_files.renderable)
        if "(none)" in current:
            lines = ["Recent files:", path]
        else:
            lines = current.splitlines()
            lines.append(path)
            lines = lines[:12]
        self.recent_files.update("\n".join(lines))
