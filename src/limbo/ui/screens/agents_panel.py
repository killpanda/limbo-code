"""Subagents panel (``/agents``): the task board, live, with direct control.

↑↓ selects a task; the pane below shows its details and the tail of its
subagent's transcript. Keys act on the selected task the way ``/task``
does (the main agent is told about every action).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Label, ListItem, ListView, Static

from limbo.subagents.events import BackgroundEvent, TaskChanged
from limbo.subagents.supervisor import STATUS_LABEL, Supervisor
from limbo.ui.screens.dialog import Dialog

_ICON = {
    "proposed": "?",
    "pending": "·",
    "ready": "○",
    "running": "●",
    "waiting": "◌",
    "paused": "Ⅱ",
    "completing": "◐",
    "done": "✓",
    "failed": "✗",
    "cancelled": "–",
}


class AgentsPanel(Dialog[str | None]):
    """Dismisses with ``@<agent> `` when the user wants to message the
    selected subagent (the main screen prefills the input), else None."""

    TITLE_TEXT = "子代理"
    HINTS = (
        ("↑↓", "选择"),
        ("p", "暂停"),
        ("r", "恢复"),
        ("c", "取消"),
        ("t", "重试"),
        ("a/x", "批准/拒绝"),
        ("m", "发消息"),
        ("esc", "关闭"),
    )
    BINDINGS = [
        *Dialog.BINDINGS,
        Binding("p", "act('pause')", show=False),
        Binding("r", "act('resume')", show=False),
        Binding("c", "act('cancel')", show=False),
        Binding("t", "act('retry')", show=False),
        Binding("a", "act('approve')", show=False),
        Binding("x", "act('reject')", show=False),
        Binding("m", "message", show=False),
    ]

    def __init__(
        self,
        supervisor: Supervisor,
        subscribe: Callable[[Callable[[BackgroundEvent], None]], Callable[[], None]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sup = supervisor
        self._subscribe = subscribe
        self._unsubscribe: Callable[[], None] | None = None
        self._ids: list[str] = []
        self._flash = ""

    def compose_body(self) -> ComposeResult:
        yield ListView(id="agents-list")
        with VerticalScroll(id="agents-detail"):
            yield Static(id="agents-detail-text", markup=False)

    def on_mount(self) -> None:
        super().on_mount()
        self._unsubscribe = self._subscribe(self._on_event)
        self._rebuild()
        self.query_one(ListView).focus()

    def on_unmount(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()

    def _on_event(self, event: BackgroundEvent) -> None:
        if isinstance(event, TaskChanged):
            if event.task.id in self._ids and len(self._ids) == len(self._sup.board.tasks):
                self._refresh_rows()
            else:
                self._rebuild()

    def _row(self, task_id: str) -> Text:
        v = self.app.current_theme.variables
        task = self._sup.board.tasks[task_id]
        text = Text(f"{_ICON.get(task.status, '·')} {task.id} ", Style(bold=True))
        text.append(f"{STATUS_LABEL.get(task.status, task.status)} ")
        text.append(f"{task.agent_name} ", Style(color=v.get("text-secondary", "")))
        text.append(task.title)
        activity = self._sup.activity(task.id)
        cp = task.last_checkpoint
        tail = activity or (cp.phase if cp is not None and not task.terminal else "")
        if task.status == "waiting":
            tail = task.wait_reason or "等待 " + ", ".join(task.waiting_on)
        if tail:
            text.append(f"  {tail}", Style(color=v.get("text-tertiary", "")))
        text.no_wrap = True
        text.overflow = "ellipsis"
        return text

    def _rebuild(self) -> None:
        view = self.query_one(ListView)
        index = view.index or 0
        self._ids = list(self._sup.board.tasks)
        view.clear()
        for task_id in self._ids:
            view.append(ListItem(Label(self._row(task_id), markup=False)))
        if self._ids:
            view.index = min(index, len(self._ids) - 1)
        self._show_detail()

    def _refresh_rows(self) -> None:
        view = self.query_one(ListView)
        for item, task_id in zip(view.children, self._ids, strict=False):
            item.query_one(Label).update(self._row(task_id))
        self._show_detail()

    def _selected(self) -> str | None:
        index = self.query_one(ListView).index
        if index is None or not (0 <= index < len(self._ids)):
            return None
        return self._ids[index]

    def _show_detail(self) -> None:
        pane = self.query_one("#agents-detail-text", Static)
        task_id = self._selected()
        if task_id is None:
            pane.update("任务板是空的。让主 agent 拆分任务，或用 /agents scan 查看项目画像。")
            return
        lines = [self._sup.status_text(task_id)]
        transcript = self._sup.transcript(task_id, limit=12)
        if transcript:
            lines.append("")
            lines.append("— 最近记录 —")
            for role, body in transcript:
                first = body.strip().splitlines()[0] if body.strip() else ""
                lines.append(f"[{role}] {first[:160]}")
        if self._flash:
            lines.insert(0, f"» {self._flash}\n")
        pane.update("\n".join(lines))

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        self._flash = ""
        self._show_detail()

    def action_act(self, action: str) -> None:
        task_id = self._selected()
        if task_id is None:
            return
        self._flash = self._sup.control(task_id, action, by="user")
        self._rebuild()

    def action_message(self) -> None:
        task_id = self._selected()
        if task_id is not None:
            self.dismiss(f"@{self._sup.board.tasks[task_id].agent_name} ")


class PlanPicker(Dialog[str | None]):
    """Suggest mode: the main agent proposed tasks. Dismisses with
    ``approve`` / ``reject``, or None to decide later (``/task``)."""

    TITLE_TEXT = "子任务计划"
    HINTS = (("↑↓", "选择"), ("enter", "确认"), ("esc", "稍后决定"))

    _CHOICES = (
        ("approve", "✓ 批准并开始"),
        ("reject", "✗ 全部拒绝"),
        (None, "… 稍后决定（/task approve <id|all>）"),
    )

    def __init__(self, supervisor: Supervisor, task_ids: list[str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._sup = supervisor
        self._task_ids = task_ids

    def compose_body(self) -> ComposeResult:
        lines = []
        for task_id in self._task_ids:
            task = self._sup.board.tasks.get(task_id)
            if task is None:
                continue
            line = f"{task.id} [{task.profile}] {task.title}"
            if task.depends_on:
                line += f"（依赖 {', '.join(task.depends_on)}）"
            lines.append(line)
            if task.brief:
                lines.append(f"    {task.brief.strip().splitlines()[0][:140]}")
        yield Static("\n".join(lines), id="plan-tasks", markup=False)
        yield ListView(*[ListItem(Label(label, markup=False)) for _, label in self._CHOICES])

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one(ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is not None and 0 <= index < len(self._CHOICES):
            self.dismiss(self._CHOICES[index][0])
