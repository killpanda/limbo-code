"""Textual App for Limbo."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.confirm import Confirmed, Rejected


class LimboApp(App[None]):
    """Main TUI application."""

    TITLE = "Limbo"
    CSS_PATH = "app.tcss"

    def __init__(
        self,
        workdir: str | Path,
        config: Config | None = None,
        llm_client: LLMClient | None = None,
        session_dir: Path | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workdir = Path(workdir).resolve()
        self.config = config
        self.llm_client = llm_client
        self.session_dir = session_dir
        theme = (config.ui.theme if config else None) or None
        if theme:
            try:
                self.theme = theme
            except Exception:  # noqa: BLE001 - unknown theme name, keep default
                pass

    def on_mount(self) -> None:
        self.push_screen(
            MainScreen(
                workdir=self.workdir,
                config=self.config,
                llm_client=self.llm_client,
                session_dir=self.session_dir,
            )
        )

    def on_confirmed(self, _event: Confirmed) -> None:
        for screen in reversed(self.screen_stack):
            if isinstance(screen, MainScreen):
                screen.handle_confirmation()
                break

    def on_rejected(self, _event: Rejected) -> None:
        for screen in reversed(self.screen_stack):
            if isinstance(screen, MainScreen):
                screen.handle_rejection()
                break
