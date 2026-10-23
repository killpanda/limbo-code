"""Top status bar: agent state on the left, model/tokens/workdir on the right.

Active states (thinking / tool) animate a braille spinner and show the
elapsed time of the current activity (RFC LIM-16 v1.1 P1-1); token usage is
updated by the main screen as ``UsageUpdate`` events arrive from the agent.

Limbo Dusk: a ``╱`` diagonal field fills the gap between the two sides. On
narrow terminals (sidebar hidden) the bar also shows the gradient wordmark
and the context summary; on wide ones the sidebar carries those, so the CSS
hides the logo and the right-hand summary.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from limbo.ui.gradient import gradient_text, ramp, theme_gradient

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_DIAG = "╱" * 400


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
        self._plan_mode = False
        self._daily_mode = False
        # NOTE: attribute names must not collide with MessagePump internals
        # (e.g. `_context` is a method on MessagePump).
        self._context_percent: int | None = None
        self._changes: tuple[int, int, int] = (0, 0, 0)
        self._logo_label = Static(id="statusbar-logo", markup=False)
        self._state_label = Static("● idle", id="statusbar-state", markup=False)
        self._diag_label = Static(_DIAG, id="statusbar-diag", markup=False)
        self._context_label = Static(self._context_text(), id="statusbar-context", markup=False)
        self._style = "idle"
        self._text = "idle"
        self._started: float | None = None
        self._frame = 0
        # Goal-mode badge: (rounds_completed, max_rounds) while a goal is
        # active, None otherwise; _goal_running toggles the rainbow.
        self._goal: tuple[int, int] | None = None
        self._goal_running = False
        self._state_listeners: list[Callable[[str, str], None]] = []

    def compose(self) -> ComposeResult:
        yield self._logo_label
        yield self._state_label
        yield self._diag_label
        yield self._context_label

    def _context_text(self) -> str:
        parts = [self._model]
        if self._context_percent is not None:
            parts.append(f"{self._context_percent}%")
        if self._code_mode:
            parts.append("⌨code")
        if self._plan_mode:
            parts.append("📋plan")
        if self._daily_mode:
            parts.append("☀daily")
        if self._tokens:
            parts.append(f"↓↑ {_format_tokens(self._tokens)}")
        files, added, removed = self._changes
        if files:
            parts.append(f"+{added}−{removed}")
        if self._queued:
            parts.append(f"⏳ {self._queued} 条排队中")
        parts.append(self._workdir)
        return " · ".join(parts)

    def _refresh_context(self) -> None:
        try:
            self._context_label.update(self._context_text())
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def on_state(self, listener: Callable[[str, str], None]) -> None:
        self._state_listeners.append(listener)

    def set_context(self, percent: int | None) -> None:
        """Context-window fill, shown in the compact (narrow) summary."""
        if percent != self._context_percent:
            self._context_percent = percent
            self._refresh_context()

    def set_changes(self, totals: tuple[int, int, int]) -> None:
        """(files, lines added, lines removed) changed this session."""
        if totals != self._changes:
            self._changes = totals
            self._refresh_context()

    def _render_logo(self) -> None:
        theme = self.app.current_theme
        text = Text("Limbo™ ", style=theme.variables.get("text-tertiary", ""))
        text.append_text(gradient_text("LIMBO", theme_gradient(theme), True))
        text.append(" ")
        try:
            self._logo_label.update(text)
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def set_queued(self, count: int) -> None:
        """Update the queued-steer-message counter (RFC LIM-20)."""
        if count == self._queued:
            return
        self._queued = count
        self._refresh_context()

    def set_tokens(self, total: int) -> None:
        """Update the cumulative token counter on the right side."""
        self._tokens = total
        self._refresh_context()

    def set_model(self, model: str) -> None:
        """Update the displayed model name (after a /model switch)."""
        self._model = model
        self._refresh_context()

    def set_code_mode(self, enabled: bool) -> None:
        """Show/hide the ⌨code badge (Code Mode presentation toggle)."""
        if enabled == self._code_mode:
            return
        self._code_mode = enabled
        self._refresh_context()

    def set_plan_mode(self, enabled: bool) -> None:
        """Show/hide the 📋plan badge (read-only plan mode)."""
        if enabled == self._plan_mode:
            return
        self._plan_mode = enabled
        try:
            self._context_label.update(self._context_text())
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def set_daily_mode(self, enabled: bool) -> None:
        """Show/hide the ☀daily badge (daily mode posture)."""
        if enabled == self._daily_mode:
            return
        self._daily_mode = enabled
        self._refresh_context()

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
        stops = theme_gradient(self.app.current_theme)
        text = Text()
        if self._goal_running:
            # The theme gradient flows through the badge while the loop runs.
            span = max(1, len(label) - 1)
            for i, ch in enumerate(label):
                t = ((i / span) + self._frame * 0.08) % 1.0
                text.append(ch, style=f"bold {ramp(stops, 1 - abs(2 * t - 1))}")
        else:
            text.append(label, style=f"bold {stops[-1]}")
        return text

    def set_state(self, text: str, style: str = "idle") -> None:
        """Update the left side. ``style`` is one of idle/thinking/tool.

        Listeners registered via ``on_state`` (the activity line) see every
        transition too, so the many call sites stay single-sourced here.
        """
        for listener in self._state_listeners:
            listener(text, style)
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
        self._render_logo()
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self._render_logo())
        # 8 fps spinner; cheap no-op while idle.
        self.set_interval(1 / 8, self._tick)

    def _tick(self) -> None:
        if self._style in ("thinking", "tool") or self._goal_running:
            self._frame += 1
            self._render_state()
