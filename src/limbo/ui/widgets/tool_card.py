"""Inline tool-call card shown in the chat flow.

A tool card renders as a single summary line (state symbol + tool name +
argument summary + elapsed time) and can be expanded to show the full tool
output. State machine: running → success | error.

The output body is lazy: it is created only when the card is expanded and
destroyed again when collapsed, so a long session's DOM stays dominated by
one-line headers instead of hundreds of `RichLog`s full of tool output.
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
# ``description`` (run_code's program intent) is last: only tools without
# a path/command/pattern/old_text hit it.
_SUMMARY_KEYS = ("path", "command", "pattern", "old_text", "description")


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
        self._body_content: list[tuple[str, str | None]] | None = None
        self._body: RichLog | None = None
        self.add_class("running")

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), classes="tool-header", markup=False)
        # The output body is NOT composed here: it is created lazily on the
        # first expansion (see _ensure_body) and destroyed on collapse, so
        # collapsed cards cost one line each in the DOM.

    def on_mount(self) -> None:
        # State transitions may have happened before composition finished
        # (events can arrive in the same loop turn as the card creation).
        self._refresh_header()

    def on_unmount(self) -> None:
        # The lazily-created body is a dynamic child: it does not survive a
        # prune/re-mount (compose only yields the header). Drop the dead
        # RichLog reference so the restored card expands cleanly on the
        # first click instead of collapsing a stale body.
        self._body = None

    @property
    def header(self) -> Static:
        return self.query_one(".tool-header", Static)

    @property
    def body(self) -> RichLog | None:
        """The lazily-created output body, or None while collapsed."""
        return self._body

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
        """Expand to show the output body, or collapse and destroy it."""
        if not self._has_body:
            return
        if self._body is None or not self._body.display:
            self._ensure_body()
        else:
            self._drop_body()

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
        parts = self._body_parts(content, lexer)
        if not parts:
            return
        self._has_body = True
        self._body_content = parts
        # Do not render yet: the body is materialized on first expansion.
        # An already-expanded card picks the new content up immediately.
        if self._body is not None and self._body.display:
            self._write_body(self._body_content)

    def _body_parts(
        self, content: str, lexer: str | None
    ) -> list[tuple[str, str | None]]:
        """Body sections, in display order.

        run_code's ``code`` argument is the program source itself — the
        thing a human wants to see to know what Code Mode did — so it is
        shown first (python-highlighted), then the tool's own result.
        """
        parts: list[tuple[str, str | None]] = []
        if self.tool_name == "run_code":
            code = self.arguments.get("code")
            if isinstance(code, str) and code:
                parts.append((code, "python"))
        if content:
            parts.append((content, lexer))
        return parts

    def _ensure_body(self) -> None:
        """Create and mount the output body (once per expansion)."""
        if self._body is None:
            body = RichLog(classes="tool-body", wrap=True, max_lines=1000)
            self.mount(body)
            self._body = body
            if self._body_content:
                self._write_body(self._body_content)
        self._body.display = True

    def _drop_body(self) -> None:
        """Destroy the output body: collapsed cards are one line in the DOM."""
        if self._body is not None:
            self._body.remove()
            self._body = None

    def _write_body(self, parts: list[tuple[str, str | None]]) -> None:
        assert self._body is not None  # only called after _ensure_body
        # A second set_success/set_error on an expanded card must not
        # duplicate earlier content in the append-only RichLog.
        self._body.clear()
        for content, lexer in parts:
            renderable: Any = Text(content)
            if lexer:
                try:
                    from rich.syntax import Syntax

                    renderable = Syntax(content, lexer, theme=self._syntax_theme())
                except Exception:  # noqa: BLE001 - fall back to plain text
                    renderable = Text(content)
            self._body.write(renderable)

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
