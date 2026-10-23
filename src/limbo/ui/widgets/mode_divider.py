"""Mode divider (Limbo Dawn): the chat line that marks a daily mode switch.

``╱╱╱╱╱╱  ☀ Daily mode · …  ╱╱╱…`` across the chat width, in the current
theme's gradient. On a manual switch a band of light sweeps across it once
(Limbo Motion "Dusk Sweep", 1.2s): left to right into daily mode, right to
left back to coding. It stays in the flow, so scrolling back shows where
the conversation changed posture.
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from textual.widgets import Static

from limbo.ui import motion
from limbo.ui.gradient import theme_gradient

SWEEP_FRAMES = 24
_LEAD = "╱" * 6


def divider_line(label: str, width: int) -> str:
    """``label`` between ``╱`` fields, padded to ``width`` cells."""
    line = f"{_LEAD}  {label}  "
    return line + "╱" * max(4, width - cell_len(line))


class ModeDivider(Static):
    def __init__(self, label: str, *, daily: bool, animate: bool, **kwargs: Any) -> None:
        super().__init__(divider_line(label, 0), classes="mode-divider", markup=False, **kwargs)
        self.label = label
        self.daily = daily
        self._sweep = animate
        self._frame = SWEEP_FRAMES

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self._render_line())
        clock = motion.get_clock(self.app)
        if self._sweep and clock.mode == motion.FULL:
            self._frame = 0
            clock.subscribe(self, self._on_frame)
        self._render_line()

    def on_unmount(self) -> None:
        try:
            motion.get_clock(self.app).unsubscribe(self)
        except Exception:  # noqa: BLE001 - app already gone
            pass

    def on_resize(self) -> None:
        self._render_line()

    def _on_frame(self, _step: int) -> None:
        self._frame += 1
        self._render_line()
        if self._frame >= SWEEP_FRAMES:
            motion.get_clock(self.app).unsubscribe(self)

    def _render_line(self) -> None:
        theme = self.app.current_theme
        variables = theme.variables
        dim = variables.get("diag", variables.get("text-tertiary", "#444444"))
        line = divider_line(self.label, self.size.width)
        self.update(
            motion.sweep_lines(
                (line,),
                theme_gradient(theme),
                dim,
                self._frame,
                total=SWEEP_FRAMES,
                mode=motion.get_clock(self.app).mode,
                reverse=not self.daily,
            )
        )
