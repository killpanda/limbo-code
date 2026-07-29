"""Top status bar: agent state on the left, model/workdir on the right."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static


class StatusBar(Horizontal):
    """One-line status bar docked at the top of the screen."""

    def __init__(self, model: str, workdir: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # NOTE: attribute names must not collide with MessagePump internals
        # (e.g. `_context` is a method on MessagePump).
        self._state_label = Static("● idle", id="statusbar-state", markup=False)
        self._context_label = Static(
            f"{model} · {workdir}", id="statusbar-context", markup=False
        )

    def compose(self) -> ComposeResult:
        yield self._state_label
        yield self._context_label

    def set_state(self, text: str, style: str = "idle") -> None:
        """Update the left side. ``style`` is one of idle/thinking/tool."""
        self._state_label.update(f"● {text}")
        self._state_label.set_class(style == "thinking", "thinking")
        self._state_label.set_class(style == "tool", "tool")
