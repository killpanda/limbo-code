"""Textual App for Limbo."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import events
from textual.app import App

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.ui.clipboard import write_clipboard_text
from limbo.ui.key_fixes import apply_key_fixes
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
        apply_key_fixes()
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

    def copy_to_clipboard(self, text: str) -> None:
        """Copy text via native OS tools first, OSC 52 as a fallback (LIM-54).

        Textual's default implementation only emits an OSC 52 escape, which
        macOS Terminal.app silently ignores — combined with Terminal.app
        never forwarding Cmd+C to the app, drag-select + copy was a no-op
        there. Native tools (``pbcopy`` etc.) work in every terminal; OSC 52
        is kept as a fallback for SSH/remote sessions where the local
        clipboard tool would target the wrong machine.

        The native write runs in a worker thread (same discipline as the
        paste path in ``InputWidget``): a synchronous ``subprocess.run`` on
        the event loop would freeze the UI for the whole PowerShell cold
        start on Windows, or up to ``_CMD_TIMEOUT`` on a hung backend.
        """
        # Keep App.clipboard in sync on the native path too (Textual only
        # sets it on its own OSC 52 path).
        self._clipboard = text
        self.run_worker(
            self._copy_to_clipboard_native(text),
            group="clipboard-copy",
            exclusive=True,
        )

    async def _copy_to_clipboard_native(self, text: str) -> None:
        if await asyncio.to_thread(write_clipboard_text, text):
            return
        super().copy_to_clipboard(text)

    def on_text_selected(self, event: events.TextSelected) -> None:
        """Copy the selection to the OS clipboard as soon as the drag ends.

        macOS Terminal.app handles Cmd+C natively (it never reaches the
        app) but has no native selection while the app captures the mouse,
        so the only way to make the select-then-Cmd+C gesture work there is
        to have already copied the text on mouse-up.
        """
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        """Fallbacks for limbo's custom CSS variables.

        Lets users switch to any Textual built-in theme via ``ui.theme``
        without breaking app.tcss (which references ``$text-tertiary`` etc.).
        The limbo themes define their own values for all of these.
        """
        return dict(LIMBO_DARK.variables)
