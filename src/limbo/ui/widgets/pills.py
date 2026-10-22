"""Status pills above the input (Limbo Dusk): queued messages and /goal.

Hidden when there is nothing to show. Each pill is a short bordered-looking
chip rendered as text: ``▸ 队列 2`` / ``◐ 目标 · 第 3/10 轮``.
"""

from __future__ import annotations

from typing import Any

from rich.style import Style
from rich.text import Text
from textual.widgets import Static


class Pills(Static):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, markup=False, **kwargs)
        self._queued = 0
        self._goal: tuple[int, int] | None = None

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self._render_pills())
        self._render_pills()

    def set_queued(self, count: int) -> None:
        if count != self._queued:
            self._queued = count
            self._render_pills()

    def set_goal(self, progress: tuple[int, int] | None) -> None:
        if progress != self._goal:
            self._goal = progress
            self._render_pills()

    @property
    def plain(self) -> str:
        return self._text().plain

    def _text(self) -> Text:
        theme = self.app.current_theme
        v = theme.variables
        mid = v.get("text-secondary", "")
        chip_bg = v.get("selected-bg", str(theme.panel))
        text = Text()

        def chip(icon: str, icon_color: str, label: str, value: str) -> None:
            if text:
                text.append("  ")
            text.append(f" {icon} ", Style(color=icon_color, bgcolor=chip_bg))
            text.append(f"{label} ", Style(color=mid, bgcolor=chip_bg))
            text.append(f"{value} ", Style(color=str(theme.accent), bgcolor=chip_bg, bold=True))

        if self._goal is not None:
            done, total = self._goal
            chip("◐", str(theme.warning), "目标", f"{done}/{total}")
        if self._queued:
            chip("▸", str(theme.primary), "排队", str(self._queued))
        return text

    def _render_pills(self) -> None:
        text = self._text()
        self.display = bool(text)
        self.update(text)
