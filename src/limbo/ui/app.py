"""Textual App for Limbo."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from limbo.ui.screens.main import MainScreen


class LimboApp(App[None]):
    """Main TUI application."""

    CSS_PATH = None

    def __init__(self, workdir: str | Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = Path(workdir).resolve()

    def on_mount(self) -> None:
        self.push_screen(MainScreen(workdir=self.workdir))
