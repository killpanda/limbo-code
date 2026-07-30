"""Textual App for Limbo."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.ui.screens.main import MainScreen
from limbo.ui.theme import BUILTIN_THEMES, DEFAULT_THEME, LIMBO_DARK


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
        resume: Path | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workdir = Path(workdir).resolve()
        self.config = config
        self.llm_client = llm_client
        self.session_dir = session_dir
        self.resume = resume
        for builtin in BUILTIN_THEMES:
            self.register_theme(builtin)
        theme_name = (config.ui.theme if config else None) or DEFAULT_THEME
        try:
            self.theme = theme_name
        except Exception:  # noqa: BLE001 - unknown theme name, use limbo default
            self.theme = DEFAULT_THEME

    def on_mount(self) -> None:
        self.push_screen(
            MainScreen(
                workdir=self.workdir,
                config=self.config,
                llm_client=self.llm_client,
                session_dir=self.session_dir,
                resume=self.resume,
            )
        )

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Fallbacks for limbo's custom CSS variables.

        Lets users switch to any Textual built-in theme via ``ui.theme``
        without breaking app.tcss (which references ``$text-tertiary`` etc.).
        The limbo themes define their own values for all of these.
        """
        return dict(LIMBO_DARK.variables)
