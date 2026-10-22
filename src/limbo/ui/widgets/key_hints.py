"""Bottom key-hint bar (Limbo Dusk): ``key`` in the foreground colour,
its description dimmed, so the line scans as a row of keys."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.widgets import Static

DEFAULT_HINTS: tuple[tuple[str, str], ...] = (
    ("enter", "发送"),
    ("shift+enter", "换行"),
    ("/", "命令"),
    ("ctrl+p", "面板"),
    ("ctrl+o", "展开"),
    ("esc", "中断"),
)


class KeyHints(Static):
    """One line of key hints; re-renders on theme change."""

    def __init__(
        self, hints: tuple[tuple[str, str], ...] = DEFAULT_HINTS, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, markup=False, **kwargs)
        self._hints = hints

    @property
    def plain(self) -> str:
        return "  ".join(f"{key} {desc}" for key, desc in self._hints)

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self._render_hints())
        self._render_hints()

    def set_hints(self, hints: tuple[tuple[str, str], ...]) -> None:
        self._hints = hints
        self._render_hints()

    def _render_hints(self) -> None:
        theme = self.app.current_theme
        key_style = f"bold {theme.foreground}"
        desc_style = theme.variables.get("text-tertiary", "")
        text = Text()
        for i, (key, desc) in enumerate(self._hints):
            if i:
                text.append("  ")
            text.append(key, style=key_style)
            text.append(" " + desc, style=desc_style)
        self.update(text)
