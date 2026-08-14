"""Top status bar: agent state on the left, model/tokens/workdir on the right.

Active states (thinking / tool) animate a braille spinner and show the
elapsed time of the current activity (RFC LIM-16 v1.1 P1-1); token usage is
updated by the main screen as ``UsageUpdate`` events arrive from the agent.
"""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⴦⦧⦇⦏"

# Rainbow palette for the goal-mode indicator (LIM-40): while a goal loop
# is executing, the badge cycles hues per frame; an active-but-idle goal
# shows a static cyan badge instead.
_RAINBOW = ("red", "orange1", "yellow1", "green1", "cyan1", "blue1", "magenta1")


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
        self._queued = 0
        self._code_mode = False
        # NOTE: attribute names must not collide with MessagePump internals
        # (e.g. `_context` is a method on MessagePump).
        self._state_label = Static("● idle", id="statusbar-state", markup=False)
        self._context_label = Static(self._context_text(), id="statusbar-context", markup=False)
        self._style = "idle"
        self._text = "idle"
        self._started: float | None = None
        self._frame = 0
        # Goal-mode badge: (rounds_completed, max_rounds) while a goal is
        # active, None otherwise; _goal_running toggles the rainbow.
        self._goal: tuple[int, int] | None = None
        self._goal_running = False

    def compose(self) -> ComposeResult:
        yield self._state_label
        yield self._context_label

    def _context_text(self) -> str:
        parts = [self._model]
        if self._code_mode:
            parts.append("⌨code")
        if self._tokens:
            parts.append(f"↓↑ {_format_tokens(self._tokens)}")
        if self._queued:
            parts.append(f"⏳ {self._queued} 条排队中")
        parts.append(self._workdir)
        return " · ".join(parts)

    def set_queued(self, count: int) -> None:
        """Update the queued-steer-message counter (RFC LIM-20)."""
        if count == self._queued:
            return
        self._queued = count
        try:
            self._context_label.update(self._context_text())
        except Exception:  # noqa: BLE001 - not composed yet
            pass

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

    def set_code_mode(self, enabled: bool) -> None:
        """Show/hide the ⌨code badge (Code Mode presentation toggle)."""
        if enabled == self._code_mode:
            return
        self._code_mode = enabled
        try:
            self._context_label.update(self._context_text())
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def set_goal(
        self, goal: tuple[int, int] | None, *, running: bool = False
    ) -> None:
        """Show/hide the goal-mode badge (LIM-40).

        ``goal`` is (rounds_completed, max_rounds); ``running=True`` turns
        on the rainbow animation (goal loop executing), ``False`` a static
        badge (goal active but idle). None hides the badge.
        """
        self._goal = goal
        self._goal_running = running and goal is not None
        self._render_state()

    def _goal_badge(self) -> Text:
        rounds, max_rounds = self._goal if self._goal is not None else (0, 0)
        label = f"🎯GOAL {rounds}/{max_rounds} "
        text = Text()
        if self._goal_running:
            for i, ch in enumerate(label):
                color = _RAINBOW[(i + self._frame) % len(_RAINBOW)]
                text.append(ch, style=f"bold {color}")
        else:
            text.append(label, style="bold cyan")
        return text

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
        label: str | Text
        if self._style in ("thinking", "tool") and self._started is not None:
            frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
            elapsed = time.monotonic() - self._started
            label = f"{frame} {self._text} {elapsed:.0f}s"
        else:
            label = f"● {self._text}"
        if self._goal is not None:
            badge = self._goal_badge()
            badge.append(label)
            label = badge
        try:
            self._state_label.update(label)
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def on_mount(self) -> None:
        # 8 fps spinner; cheap no-op while idle.
        self.set_interval(1 / 8, self._tick)

    def _tick(self) -> None:
        if self._style in ("thinking", "tool") or self._goal_running:
            self._frame += 1
            self._render_state()
