"""ctrl+p command palette (Limbo Dusk): one fuzzy list over three tabs.

Tabs are 系统 (built-in slash commands), 会话 (history sessions) and 技能
(skills); ``tab`` cycles them and typing filters the active one. Enter
dismisses with a :class:`PaletteChoice`; the main screen decides what to do
(run a command, prefill one that takes arguments, switch session).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from limbo.sessions import SessionMeta
from limbo.skills import Skill
from limbo.ui.commands import SlashCommand
from limbo.ui.screens.dialog import Dialog
from limbo.ui.widgets.landing import relative_time

TABS = (("system", "系统"), ("sessions", "会话"), ("skills", "技能"))


@dataclass(frozen=True)
class PaletteChoice:
    kind: str  # "command" | "session" | "skill"
    value: str  # command name / session path / "/skill-name"
    takes_args: bool = False


@dataclass(frozen=True)
class _Entry:
    choice: PaletteChoice
    title: str
    detail: str


def fuzzy_score(query: str, text: str) -> int | None:
    """Subsequence match score (higher is better); None when it doesn't match."""
    if not query:
        return 0
    query, text = query.lower(), text.lower()
    if query in text:
        return 1000 - text.index(query) * 2 - len(text)
    pos, score, last = 0, 0, -2
    for ch in query:
        found = text.find(ch, pos)
        if found < 0:
            return None
        score += 10 if found == last + 1 else 1
        last, pos = found, found + 1
    return score - len(text)


class CommandPalette(Dialog[PaletteChoice | None]):
    TITLE_TEXT = "命令"
    HINTS = (("tab", "切换分组"), ("↑↓", "选择"), ("enter", "执行"), ("esc", "关闭"))
    BINDINGS = [
        Binding("escape", "cancel", "取消", show=False),
        Binding("tab", "next_tab", show=False, priority=True),
        Binding("shift+tab", "prev_tab", show=False, priority=True),
        Binding("down", "cursor(1)", show=False, priority=True),
        Binding("up", "cursor(-1)", show=False, priority=True),
    ]

    def __init__(
        self,
        commands: list[SlashCommand],
        sessions: list[SessionMeta],
        skills: list[Skill],
        current_session: str | None = None,
    ) -> None:
        super().__init__()
        self._tab = 0
        self._entries: dict[str, list[_Entry]] = {
            "system": [
                _Entry(PaletteChoice("command", c.name, c.takes_args), c.name, c.description)
                for c in commands
                if c.kind == "builtin"
            ],
            "sessions": [
                _Entry(
                    PaletteChoice("session", str(meta.path)),
                    meta.title or "（无标题）",
                    relative_time(meta.updated_at)
                    + (" · 当前" if meta.id == current_session else ""),
                )
                for meta in sessions
            ],
            "skills": [
                _Entry(PaletteChoice("skill", f"/{s.name}", True), f"/{s.name}", s.description)
                for s in skills
            ],
        }
        self._shown: list[_Entry] = []

    def compose_body(self) -> ComposeResult:
        yield Static(classes="palette-tabs", markup=False)
        yield Input(placeholder="输入以筛选…", classes="palette-input")
        yield OptionList(classes="palette-list")

    def on_mount(self) -> None:
        super().on_mount()
        self._render_tabs()
        self._refilter()
        self.query_one(Input).focus()

    @property
    def tab(self) -> str:
        return TABS[self._tab][0]

    @property
    def shown(self) -> list[PaletteChoice]:
        return [e.choice for e in self._shown]

    def _render_frame(self) -> None:
        super()._render_frame()
        if self.is_mounted:
            self._render_tabs()

    def _render_tabs(self) -> None:
        theme = self.app.current_theme
        v = theme.variables
        text = Text()
        for i, (key, label) in enumerate(TABS):
            count = len(self._entries[key])
            if i:
                text.append("  ")
            if i == self._tab:
                style = Style(
                    color=v.get("selected-fg", str(theme.foreground)),
                    bgcolor=v.get("selected-bg", str(theme.panel)),
                    bold=True,
                )
            else:
                style = Style(color=v.get("text-tertiary", ""))
            text.append(f" {label} {count} ", style)
        self.query_one(".palette-tabs", Static).update(text)

    def _refilter(self) -> None:
        query = self.query_one(Input).value.strip()
        scored = []
        for n, entry in enumerate(self._entries[self.tab]):
            score = fuzzy_score(query, f"{entry.title} {entry.detail}")
            if score is not None:
                scored.append((-score if query else n, n, entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        self._shown = [entry for _, _, entry in scored]

        theme = self.app.current_theme
        v = theme.variables
        options = self.query_one(OptionList)
        options.clear_options()
        if not self._shown:
            options.add_option(Option(Text("没有匹配项", Style(color=v.get("text-tertiary", ""))),
                                      disabled=True))
            return
        width = max(len(e.title) for e in self._shown)
        for entry in self._shown:
            prompt = Text(entry.title.ljust(min(width, 28)), Style(bold=True))
            prompt.append("  " + entry.detail, Style(color=v.get("text-secondary", "")))
            prompt.no_wrap = True
            prompt.overflow = "ellipsis"
            options.add_option(Option(prompt))
        options.highlighted = 0

    def on_input_changed(self, _event: Input.Changed) -> None:
        self._refilter()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        self._choose(self.query_one(OptionList).highlighted)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._choose(event.option_index)

    def _choose(self, index: int | None) -> None:
        if index is not None and 0 <= index < len(self._shown):
            self.dismiss(self._shown[index].choice)

    def _switch(self, delta: int) -> None:
        self._tab = (self._tab + delta) % len(TABS)
        self._render_tabs()
        self._refilter()

    def action_next_tab(self) -> None:
        self._switch(1)

    def action_prev_tab(self) -> None:
        self._switch(-1)

    def action_cursor(self, delta: int) -> None:
        options = self.query_one(OptionList)
        if not self._shown:
            return
        current = options.highlighted or 0
        options.highlighted = max(0, min(len(self._shown) - 1, current + delta))


def session_path(choice: PaletteChoice) -> Path:
    return Path(choice.value)
