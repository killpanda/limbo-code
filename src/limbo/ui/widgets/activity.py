"""Activity line above the input (Limbo Motion "Tide").

Shown only while the agent is busy: a row of ``╱`` with a bright crest
sweeping through, the activity label with an animated ellipsis, elapsed
seconds and the interrupt hint. Thinking scrambles the crest glyphs; tool
runs and verification use the calmer light-only variant so nobody reads
"the model is thinking" into a shell command.
"""

from __future__ import annotations

import time
from typing import Any

from rich.style import Style
from textual.widgets import Static

from limbo.ui import motion
from limbo.ui.gradient import theme_gradient
from limbo.ui.reflow import update_line

_LABELS = {
    "thinking…": "思考中",
    "verifying…": "验收中",
    "compacting…": "压缩上下文",
}


def activity_label(text: str) -> str:
    """Status-bar wording → the activity line's label."""
    if text in _LABELS:
        return _LABELS[text]
    if text.startswith("running ") and text.endswith("…"):
        return "运行 " + text.removeprefix("running ").removesuffix("…")
    return text.removesuffix("…")


class ActivityLine(Static):
    """Tide + label while busy; hidden when idle."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, markup=False, **kwargs)
        self._style = "idle"
        self._label = ""
        self._started: float | None = None
        self._birth = 0
        self._step = 0
        self._seed = 0

    @property
    def active(self) -> bool:
        return self._style in ("thinking", "tool")

    def on_mount(self) -> None:
        self.display = False

    def on_unmount(self) -> None:
        try:
            motion.get_clock(self.app).unsubscribe(self)
        except Exception:  # noqa: BLE001 - app already gone
            pass

    def set_state(self, text: str, style: str) -> None:
        was_active = self.active
        if style != self._style or not was_active:
            self._started = time.monotonic()
        self._style = style
        self._label = activity_label(text)
        clock = motion.get_clock(self.app)
        if self.active:
            if not was_active:
                self._birth = clock.step
                self._seed += 1
                clock.subscribe(self, self._on_frame)
            self.display = True
            self._render_line()
        else:
            clock.unsubscribe(self)
            self.display = False

    def _on_frame(self, step: int) -> None:
        self._step = step
        self._render_line()

    def _render_line(self) -> None:
        theme = self.app.current_theme
        clock = motion.get_clock(self.app)
        dim = theme.variables.get("diag", theme.variables.get("text-tertiary", "#444444"))
        tertiary = theme.variables.get("text-tertiary", "")
        secondary = theme.variables.get("text-secondary", "")
        line = motion.tide_frame(
            self._step,
            theme_gradient(theme),
            dim,
            mode=clock.mode,
            scramble=self._style == "thinking",
            seed=self._seed,
            birth=self._birth if clock.mode == motion.FULL else None,
        )
        line.append(" ")
        line.append(self._label + motion.ellipsis(self._step, clock.mode), Style(color=secondary))
        elapsed = 0.0 if self._started is None else time.monotonic() - self._started
        line.append(f"  · {elapsed:.0f}s · esc 中断", Style(color=tertiary))
        update_line(self, line)

    @property
    def label(self) -> str:
        return self._label if self.active else ""
