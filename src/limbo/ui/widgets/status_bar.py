"""Top status bar: agent state on the left, model/tokens/workdir on the right.

Active states (thinking / tool) animate a braille spinner and show the
elapsed time of the current activity (RFC LIM-16 v1.1 P1-1); token usage is
updated by the main screen as ``UsageUpdate`` events arrive from the agent.
"""

from __future__ import annotations

import time
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⴦⦧⦇⦏"


def _format_tokens(total: int) -> str:
    if total >= 1000:
        return f"{total / 1000:.1f}k"
    return str(total)


class StatusBar(Horizontal):
    """One-line status bar docked at the top of the screen."""

    def __init__(self, model: str, workdir: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._workdir = workdir
        self._tokens = 0
        # NOTE: attribute names must not collide with MessagePump internals
        # (e.g. `_context` is a method on MessagePump).
        self._state_label = Static("● idle", id="statusbar-state", markup=False)
        self._context_label = Static(self._context_text(), id="statusbar-context", markup=False)
        self._style = "idle"
        self._text = "idle"
        self._started: float | None = None
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield self._state_label
        yield self._context_label

    def _context_text(self) -> str:
        parts = [self._model]
        if self._tokens:
            parts.append(f"↓↑ {_format_tokens(self._tokens)}")
        parts.append(self._workdir)
        return " · ".join(parts)

    def set_tokens(self, total: int) -> None:
        """Update the cumulative token counter on the right side."""
        self._tokens = total
        try:
            self._context_label.update(self._context_text())
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def set_model(self, model: str) -> None:
        """Update the displayed model name (after a /model switch)."""
        self._model = model
        try:
            self._context_label.update(self._context_text())
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def set_state(self, text: str, style: str = "idle") -> None:
        """Update the left side. ``style`` is one of idle/thinking/tool."""
        active = style in ("thinking", "tool")
        # Reset the clock when the activity kind changes or a new one starts.
        if active and (self._style != style or self._started is None):
            self._started = time.monotonic()
        if not active:
            self._started = None
        self._style = style
        self._text = text
        self._state_label.set_class(style == "thinking", "thinking")
        self._state_label.set_class(style == "tool", "tool")
        self._render_state()

    def _render_state(self) -> None:
        if self._style in ("thinking", "tool") and self._started is not None:
            frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
            elapsed = time.monotonic() - self._started
            label = f"{frame} {self._text} {elapsed:.0f}s"
        else:
            label = f"● {self._text}"
        try:
            self._state_label.update(label)
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def on_mount(self) -> None:
        # 8 fps spinner; cheap no-op while idle.
        self.set_interval(1 / 8, self._tick)

    def _tick(self) -> None:
        if self._style in ("thinking", "tool"):
            self._frame += 1
            self._render_state()
