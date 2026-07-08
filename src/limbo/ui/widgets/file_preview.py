"""Right panel for previewing read/bash output."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class FilePreviewWidget(Static):
    """Right-side preview panel."""

    DEFAULT_CSS = """
    FilePreviewWidget {
        width: 1fr;
        height: 1fr;
        padding: 1;
    }
    """

    def show(self, title: str, content: str) -> None:
        # Render as a Text object so untrusted tool output is not parsed as Rich markup.
        text = Text(title, style="#888888")
        text.append("\n")
        text.append(content)
        self.update(text)
