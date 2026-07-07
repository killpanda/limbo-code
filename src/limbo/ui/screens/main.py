"""Main screen with three-column layout."""

from __future__ import annotations

from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.file_preview import FilePreviewWidget
from limbo.ui.widgets.input import InputWidget
from limbo.ui.widgets.sidebar import SidebarWidget


class MainScreen(Screen[None]):
    """Main three-column layout."""

    def __init__(self, workdir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = workdir

    def compose(self):
        with Horizontal():
            with Vertical(id="sidebar"):
                yield SidebarWidget(id="sidebar")
            with Vertical(id="chat-container"):
                yield ChatWidget(id="chat")
                yield InputWidget(id="input")
            with Vertical(id="preview-container"):
                yield FilePreviewWidget(id="preview")
