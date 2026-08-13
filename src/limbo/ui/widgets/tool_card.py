"""Inline tool-call card shown in the chat flow.

A tool card renders as a single summary line (state symbol + tool name +
argument summary + elapsed time) and can be expanded to show the full tool
output. State machine: running → success | error.
"""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import RichLog, Static

_STATE_SYMBOLS = {
    "running": "…",
    "success": "✓",
    "error": "✗",
    "cancelled": "⏹",
}
_STATE_LABELS = {
    "running": "运行中",
    "success": "",
    "error": "失败",
    "cancelled": "已打断",
}

# Argument keys worth showing in the one-line summary, in priority order.
_SUMMARY_KEYS = ("path", "command", "pattern", "old_text")


class ToolCard(Vertical):
    """One-line tool summary that expands to show full output."""

    def __init__(
        self,
        tool_id: str,
        name: str,
        arguments: dict[str, Any],
        *args: Any,
        agent_owned: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tool_id = tool_id
        self.tool_name = name
        self.arguments = arguments
        # Agent-owned cards back LLM tool calls: a turn interrupt means no
        # result will ever arrive, so cancel_running_tool_cards() marks them
        # cancelled. Non-agent-owned cards (user bang commands) keep running
        # — their result always arrives when the process exits.
        self.agent_owned = agent_owned
        self.state = "running"
        self._started = time.monotonic()
        self._elapsed: float | None = None
        self._has_body = False
        self._body_content: tuple[str, str | None] | None = None
        self.add_class("running")

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), classes="tool-header", markup=False)
        body = RichLog(classes="tool-body", wrap=True, max_lines=1000)
        body.display = False
        yield body

    def on_mount(self) -> None:
        # State transitions may have happened before composition finished
        # (events can arrive in the same loop turn as the card creation).
        self._refresh_header()
        if self._body_content is not None:
            self._write_body(*self._body_content)

    @property
    def header(self) -> Static:
        return self.query_one(".tool-header", Static)

    @property
    def body(self) -> RichLog:
        return self.query_one(".tool-body", RichLog)

    # -- state transitions -------------------------------------------------

    def set_success(self, output: str) -> None:
        self._set_state("success")
        self._set_body(output, lexer=self._lexer_for_body())

    def set_error(self, error: str) -> None:
        self._set_state("error")
        self._set_body(error)

    def set_cancelled(self) -> None:
        """Mark the call as interrupted by the user (RFC LIM-53).

        No result will ever arrive for this call (the stream was cut
        mid-tool-call), so there is no body to show.
        """
        self._set_state("cancelled")

    # -- expansion ----------------------------------------------------------

    def toggle(self) -> None:
        if self._has_body:
            self.body.display = not self.body.display

    def on_click(self) -> None:
        self.toggle()

    # -- internals ----------------------------------------------------------

    def _set_state(self, state: str) -> None:
        if self.state != state:
            self.remove_class(self.state)
            self.state = state
            self.add_class(state)
        if state != "running" and self._elapsed is None:
            self._elapsed = time.monotonic() - self._started
        self._refresh_header()

    def _refresh_header(self) -> None:
        try:
            self.header.update(self._header_text())
        except NoMatches:
            pass  # Not composed yet; on_mount refreshes.

    def _set_body(self, content: str, lexer: str | None = None) -> None:
        if not content:
            return
        self._has_body = True
        self._body_content = (content, lexer)
        try:
            self._write_body(content, lexer)
        except NoMatches:
            pass  # Not composed yet; on_mount writes the body.

    def _write_body(self, content: str, lexer: str | None = None) -> None:
        renderable: Any = Text(content)
        if lexer:
            try:
                from rich.syntax import Syntax

                renderable = Syntax(content, lexer, theme=self._syntax_theme())
            except Exception:  # noqa: BLE001 - fall back to plain text
                renderable = Text(content)
        self.body.write(renderable)

    def _syntax_theme(self) -> Any:
        """Match the syntax-highlighting palette to the active UI theme.

        Returns a Pygments ``Style`` subclass; rich accepts it at runtime
        though its type hints only declare ``str | SyntaxTheme``.
        """
        from limbo.ui.syntax import LimboDarkStyle, LimboLightStyle

        try:
            dark = self.app.current_theme.dark
        except Exception:  # noqa: BLE001 - no app/theme context (tests)
            dark = True
        return LimboDarkStyle if dark else LimboLightStyle

    def _lexer_for_body(self) -> str | None:
        """Pick a syntax-highlighting lexer based on the tool and its target."""
        if self.tool_name == "edit":
            return "diff"
        if self.tool_name in ("read", "write"):
            path = self.arguments.get("path")
            if isinstance(path, str) and path:
                try:
                    from pygments.lexers import get_lexer_for_filename
                    from pygments.util import ClassNotFound

                    try:
                        aliases = get_lexer_for_filename(path).aliases
                        return aliases[0] if aliases else None
                    except ClassNotFound:
                        return None
                except Exception:  # noqa: BLE001
                    return None
        return None

    def _summary(self) -> str:
        for key in _SUMMARY_KEYS:
            value = self.arguments.get(key)
            if isinstance(value, str) and value:
                first_line = value.splitlines()[0] if value.strip() else value
                return (
                    first_line if len(first_line) <= 60 else first_line[:57] + "..."
                )
        return ""

    def _header_text(self) -> str:
        symbol = _STATE_SYMBOLS.get(self.state, "?")
        parts = [f"{symbol} {self.tool_name}"]
        summary = self._summary()
        if summary:
            parts.append(summary)
        label = _STATE_LABELS.get(self.state, "")
        if label:
            parts.append(f"({label})")
        if self._elapsed is not None and self._elapsed >= 0.05:
            parts.append(f"{self._elapsed:.1f}s")
        return "  ".join(parts)
