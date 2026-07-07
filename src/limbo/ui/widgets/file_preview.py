"""Right panel for previewing read/bash output."""

from __future__ import annotations

from rich.markup import escape
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
        # Escape untrusted tool output so it is not interpreted as Rich markup.
        self.update(f"[#888888]{title}[/#888888]\n{escape(content)}")
