"""Sidebar showing session info and recent files."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.title = Static("Limbo", id="session-title")
        self.recent_files = Static("Recent files:\n(none)", id="recent-files")
        self.status = Static("Idle", id="tool-status")
        self._recent_files: list[str] = []

    def compose(self) -> ComposeResult:
        yield self.title
        yield self.recent_files
        yield self.status

    def set_status(self, text: str) -> None:
        self.status.update(text)

    def add_recent_file(self, path: str) -> None:
        if path not in self._recent_files:
            self._recent_files.append(path)
            self._recent_files = self._recent_files[:10]
        if self._recent_files:
            lines = ["Recent files:"] + self._recent_files
        else:
            lines = ["Recent files:", "(none)"]
        self.recent_files.update("\n".join(lines))
