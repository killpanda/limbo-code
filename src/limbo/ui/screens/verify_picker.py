"""Verify-command confirmation picker (LIM-40 M2).

Shown after the proposal round: the model's candidate acceptance commands
plus "edit it myself" and "skip" options. Dismisses with the chosen command
(string), the sentinel ``EDIT`` (user wants to type/edit a command), or
None (skip / cancelled — goal stays in single-turn mode without a gate).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Label, ListItem, ListView, Static

EDIT = "__edit__"


class VerifyPicker(Screen[str | None]):
    """Modal list of proposed verify commands; Enter picks, Esc skips."""

    BINDINGS = [
        Binding("escape", "cancel", "跳过"),
    ]

    def __init__(self, proposals: list[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proposals = proposals
        # Entry per ListView row: a command string, EDIT, or None (skip).
        self._entries: list[str | None] = []

    def compose(self) -> ComposeResult:
        items: list[ListItem] = []
        for command in self._proposals:
            items.append(ListItem(Label(f"✓ {command}", markup=False)))
            self._entries.append(command)
        items.append(ListItem(Label("✏️ 自己输入/编辑命令…", markup=False)))
        self._entries.append(EDIT)
        items.append(ListItem(Label("⏭️ 跳过（不自动验收）", markup=False)))
        self._entries.append(None)
        with Vertical(id="verify-picker"):
            title = "模型建议的验收方式 · Enter 确认 · Esc 跳过"
            if not self._proposals:
                title = "未能解析验收提议 · 请手动输入或跳过 · Esc 跳过"
            yield Static(title, id="picker-title", markup=False)
            yield ListView(*items)

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is None or not (0 <= index < len(self._entries)):
            return
        self.dismiss(self._entries[index])

    def action_cancel(self) -> None:
        self.dismiss(None)
