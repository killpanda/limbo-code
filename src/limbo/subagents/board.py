"""Task board: the durable DAG of delegated work (subagent design §4).

The board is event-sourced. Every change is one ``task`` record — the
same dict is applied in memory (:meth:`TaskBoard.apply`) and appended to
the main session file through the journal sink, so loading a session is
just folding its records again. There is exactly one code path for "what
a record means", live and on load.

Two kinds of edges (pi-durable's task graph):

- **owner** — who created whom. Cancellation flows *down* owner edges,
  and an owner finishes only after its owned work drained (``completing``).
- **depends_on / waiting_on** — who needs whose result. These edges only
  carry outcomes: a failed or cancelled dependency *notifies* the waiting
  task (``on_dep_fail="notify"``) unless the task asked to be cascaded.

Scheduling decisions are a pure function of the board
(:func:`reconcile`), so every rule here is unit-testable without an LLM.
Non-UI layer: never imports textual.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskStatus = Literal[
    "proposed",
    "pending",
    "ready",
    "running",
    "waiting",
    "paused",
    "completing",
    "done",
    "failed",
    "cancelled",
]
TERMINAL: frozenset[str] = frozenset({"done", "failed", "cancelled"})
# Statuses that hold a concurrency slot.
ACTIVE: frozenset[str] = frozenset({"running"})
JoinPolicy = Literal["all_settled", "fail_fast"]
DepFailPolicy = Literal["notify", "cascade"]
Isolation = Literal["shared", "worktree"]
CreatedBy = Literal["main", "user", "subagent", "auto"]

# A wait on the main agent (``ask_main``) is a pseudo-dependency
# ``q:<question id>``, resolved when the question is answered.
QUESTION_PREFIX = "q:"

TASK_RECORD_TYPE = "task"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BoardError(Exception):
    """A board verb was refused (unknown task, bad transition, cycle …)."""


class Checkpoint(BaseModel):
    seq: int
    phase: str
    summary: str = ""
    files: list[str] = Field(default_factory=list)
    progress: float | None = None
    auto: bool = False
    at: str = ""


class Outcome(BaseModel):
    status: Literal["done", "failed", "cancelled"]
    summary: str = ""
    files_changed: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    reason: str = ""


class Question(BaseModel):
    id: str
    text: str
    answer: str | None = None
    at: str = ""


class Task(BaseModel):
    """One node of the board (design §4.2)."""

    id: str
    title: str
    brief: str = ""
    acceptance: str = ""
    profile: str = "worker"
    agent_name: str = ""
    owner: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    join: JoinPolicy = "all_settled"
    on_dep_fail: DepFailPolicy = "notify"
    priority: int = 0
    scope_paths: list[str] = Field(default_factory=list)
    isolation: Isolation = "shared"
    # brief: the subagent starts from the task text only; fork: with a
    # copy of the main conversation.
    context: Literal["brief", "fork"] = "brief"
    status: TaskStatus = "pending"
    waiting_on: list[str] = Field(default_factory=list)
    wait_reason: str = ""
    session_file: str | None = None
    worktree: str | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    outcome: Outcome | None = None
    # What the subagent is told when it (re)starts: dependency outcomes,
    # the user's resume note, messages sent while it was not running.
    resume_note: str = ""
    inbox: list[str] = Field(default_factory=list)
    paused_reason: str = ""
    # The status a pause interrupted (resume goes back to it).
    paused_from: str = ""
    attempt: int = 0
    tokens: int = 0
    max_tokens: int = 0
    created_by: CreatedBy = "main"
    created_at: str = ""
    updated_at: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def last_checkpoint(self) -> Checkpoint | None:
        return self.checkpoints[-1] if self.checkpoints else None

    def open_question(self) -> Question | None:
        return next((q for q in self.questions if q.answer is None), None)


# -- records ----------------------------------------------------------------


def record(op: str, **fields: Any) -> dict[str, Any]:
    return {"type": TASK_RECORD_TYPE, "op": op, **fields}


class TaskBoard:
    """The live board: state + verbs. Every verb emits records via ``sink``.

    ``sink`` receives each record after it was applied in memory (the
    supervisor wires it to the session journal). Records are plain JSON
    dicts: ``created`` (a whole task), ``update`` (a field patch),
    ``checkpoint``, ``question``, ``answer``, ``reported`` (notice ids
    delivered to the main agent) and ``snapshot`` (the whole board, written
    after a journal rewrite dropped the event history).
    """

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.tasks: dict[str, Task] = {}
        self.reported: set[str] = set()
        self._sink = sink
        self._seq = 0

    # -- event sourcing ------------------------------------------------------

    def set_sink(self, sink: Callable[[dict[str, Any]], None] | None) -> None:
        self._sink = sink

    def _emit(self, rec: dict[str, Any]) -> None:
        self.apply(rec)
        if self._sink is not None:
            self._sink(rec)

    def apply(self, rec: dict[str, Any]) -> None:
        """Fold one record into memory (live and on load). Tolerant: a
        record that no longer makes sense (unknown task) is skipped."""
        op = rec.get("op")
        if op == "snapshot":
            self.tasks = {}
            for raw in rec.get("tasks") or []:
                try:
                    restored = Task(**raw)
                except ValueError:
                    continue
                self.tasks[restored.id] = restored
            self.reported = set(rec.get("reported") or [])
            self._seq = max(self._seq, int(rec.get("seq") or 0))
            return
        if op == "created":
            try:
                created = Task(**(rec.get("task") or {}))
            except ValueError:
                return
            self.tasks[created.id] = created
            self._bump_seq(created.id)
            return
        if op == "reported":
            self.reported.update(rec.get("ids") or [])
            return
        found = self.tasks.get(str(rec.get("id")))
        if found is None:
            return
        task = found
        if op == "update":
            fields = rec.get("fields") or {}
            try:
                updated = task.model_copy(update=_coerce(fields))
                # Validate the patched model (e.g. an unknown status).
                self.tasks[task.id] = Task(**updated.model_dump())
            except ValueError:
                return
        elif op == "checkpoint":
            try:
                task.checkpoints.append(Checkpoint(**(rec.get("checkpoint") or {})))
            except ValueError:
                return
        elif op == "question":
            try:
                task.questions.append(Question(**(rec.get("question") or {})))
            except ValueError:
                return
        elif op == "answer":
            qid = rec.get("qid")
            for q in task.questions:
                if q.id == qid:
                    q.answer = str(rec.get("answer") or "")
        elif op == "usage":
            task.tokens = int(rec.get("tokens") or 0)

    def snapshot(self) -> dict[str, Any]:
        return record(
            "snapshot",
            tasks=[t.model_dump() for t in self.tasks.values()],
            reported=sorted(self.reported),
            seq=self._seq,
        )

    def _bump_seq(self, task_id: str) -> None:
        if task_id.startswith("t") and task_id[1:].isdigit():
            self._seq = max(self._seq, int(task_id[1:]))

    # -- queries ---------------------------------------------------------------

    def get(self, task_id: str) -> Task:
        task = self.tasks.get(task_id)
        if task is None:
            raise BoardError(f"没有任务 {task_id}")
        return task

    def find(self, ref: str) -> Task | None:
        """A task by id or by agent name (``@name`` routing)."""
        ref = ref.strip().removeprefix("@")
        if ref in self.tasks:
            return self.tasks[ref]
        matches = [t for t in self.tasks.values() if t.agent_name == ref]
        # Prefer a live task when an agent name was reused.
        live = [t for t in matches if not t.terminal]
        candidates = live or matches
        return candidates[-1] if candidates else None

    def live(self) -> list[Task]:
        return [t for t in self.tasks.values() if not t.terminal]

    def with_status(self, *statuses: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.status in statuses]

    def owned_by(self, task_id: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.owner == task_id]

    def dependents(self, task_id: str) -> list[Task]:
        return [
            t for t in self.tasks.values() if task_id in t.depends_on or task_id in t.waiting_on
        ]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tasks.values():
            out[t.status] = out.get(t.status, 0) + 1
        return out

    # -- graph -----------------------------------------------------------------

    def _edges(self, task: Task) -> list[str]:
        return [d for d in [*task.depends_on, *task.waiting_on] if d in self.tasks]

    def find_cycle(self, start: str, targets: Iterable[str]) -> list[str] | None:
        """The path ``start → … → start`` that adding ``start → targets``
        would close, or None when the edges keep the graph acyclic."""
        for target in targets:
            path = self._path(target, start, set())
            if path is not None:
                return [start, *path]
        return None

    def _path(self, src: str, dst: str, seen: set[str]) -> list[str] | None:
        if src == dst:
            return [dst]
        if src in seen or src not in self.tasks:
            return None
        seen.add(src)
        for nxt in self._edges(self.tasks[src]):
            rest = self._path(nxt, dst, seen)
            if rest is not None:
                return [src, *rest]
        return None

    def critical_path(self, task_id: str) -> int:
        """How many live tasks transitively depend on ``task_id``."""
        seen: set[str] = set()
        stack = [task_id]
        while stack:
            current = stack.pop()
            for t in self.dependents(current):
                if t.id not in seen and not t.terminal:
                    seen.add(t.id)
                    stack.append(t.id)
        return len(seen)

    # -- verbs -----------------------------------------------------------------

    def next_id(self) -> str:
        self._seq += 1
        return f"t{self._seq}"

    def create(
        self,
        *,
        title: str,
        brief: str = "",
        profile: str = "worker",
        agent_name: str = "",
        depends_on: Iterable[str] = (),
        status: TaskStatus = "pending",
        created_by: CreatedBy = "main",
        **fields: Any,
    ) -> Task:
        deps = list(dict.fromkeys(depends_on))
        for dep in deps:
            if dep not in self.tasks:
                raise BoardError(f"依赖的任务 {dep} 不存在")
        task_id = self.next_id()
        task = Task(
            id=task_id,
            title=title.strip() or task_id,
            brief=brief,
            profile=profile,
            agent_name=agent_name or self._agent_name(profile),
            depends_on=deps,
            status=status,
            created_by=created_by,
            created_at=_now(),
            updated_at=_now(),
            **fields,
        )
        self._emit(record("created", task=task.model_dump()))
        return self.tasks[task_id]

    def _agent_name(self, profile: str) -> str:
        used = {t.agent_name for t in self.tasks.values()}
        n = 1
        while f"{profile}-{n}" in used:
            n += 1
        return f"{profile}-{n}"

    def update(self, task_id: str, **fields: Any) -> Task:
        self.get(task_id)
        fields["updated_at"] = _now()
        self._emit(record("update", id=task_id, fields=_dump(fields)))
        return self.tasks[task_id]

    def set_status(self, task_id: str, status: TaskStatus, **fields: Any) -> Task:
        task = self.get(task_id)
        if task.terminal and status not in TERMINAL and status != "ready":
            raise BoardError(f"任务 {task_id} 已结束（{task.status}）")
        return self.update(task_id, status=status, **fields)

    def add_checkpoint(
        self,
        task_id: str,
        phase: str,
        summary: str = "",
        files: Iterable[str] = (),
        progress: float | None = None,
        *,
        auto: bool = False,
    ) -> Checkpoint:
        task = self.get(task_id)
        cp = Checkpoint(
            seq=len(task.checkpoints) + 1,
            phase=phase,
            summary=summary,
            files=list(files),
            progress=progress,
            auto=auto,
            at=_now(),
        )
        self._emit(record("checkpoint", id=task_id, checkpoint=cp.model_dump()))
        return cp

    def wait_on(
        self, task_id: str, on: Iterable[str], reason: str = "", join: JoinPolicy | None = None
    ) -> Task:
        task = self.get(task_id)
        targets = [d for d in dict.fromkeys(on) if d != task_id]
        for dep in targets:
            if dep not in self.tasks:
                raise BoardError(f"任务 {dep} 不存在")
        cycle = self.find_cycle(task_id, targets)
        if cycle is not None:
            raise BoardError("会形成循环依赖：" + " → ".join(cycle))
        fields: dict[str, Any] = {
            "status": "waiting",
            "waiting_on": [*task.waiting_on, *[t for t in targets if t not in task.waiting_on]],
            "wait_reason": reason,
        }
        if join is not None:
            fields["join"] = join
        return self.update(task_id, **fields)

    def ask(self, task_id: str, text: str) -> Question:
        task = self.get(task_id)
        q = Question(id=f"q{len(task.questions) + 1}-{task_id}", text=text, at=_now())
        self._emit(record("question", id=task_id, question=q.model_dump()))
        self.update(
            task_id,
            status="waiting",
            waiting_on=[*task.waiting_on, QUESTION_PREFIX + q.id],
            wait_reason=f"等待主 agent 回答：{text[:80]}",
        )
        return q

    def answer(self, question_id: str, text: str) -> Task:
        for task in self.tasks.values():
            for q in task.questions:
                if q.id == question_id:
                    if q.answer is not None:
                        raise BoardError(f"问题 {question_id} 已经回答过")
                    self._emit(record("answer", id=task.id, qid=q.id, answer=text))
                    return self.tasks[task.id]
        raise BoardError(f"没有问题 {question_id}")

    def mark_reported(self, ids: Iterable[str]) -> None:
        fresh = [i for i in ids if i not in self.reported]
        if fresh:
            self._emit(record("reported", ids=fresh))

    def record_usage(self, task_id: str, tokens: int) -> None:
        task = self.tasks.get(task_id)
        if task is not None and tokens != task.tokens:
            self._emit(record("usage", id=task_id, tokens=tokens))


def _dump(fields: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, BaseModel):
            out[key] = value.model_dump()
        elif isinstance(value, list):
            out[key] = [v.model_dump() if isinstance(v, BaseModel) else v for v in value]
        else:
            out[key] = value
    return out


def _coerce(fields: dict[str, Any]) -> dict[str, Any]:
    out = dict(fields)
    if isinstance(out.get("outcome"), dict):
        out["outcome"] = Outcome(**out["outcome"])
    return out


# -- reconcile ----------------------------------------------------------------


@dataclass(frozen=True)
class Start:
    """Give ``task_id`` a slot and (re)start its subagent."""

    task_id: str


@dataclass(frozen=True)
class MakeReady:
    """Deps/waits settled: the task becomes ready with ``note``."""

    task_id: str
    note: str


@dataclass(frozen=True)
class CancelTask:
    """Cancel ``task_id`` (owner cascade or ``on_dep_fail=cascade``)."""

    task_id: str
    reason: str


@dataclass(frozen=True)
class Finalize:
    """A ``completing`` task's owned work drained: it becomes terminal."""

    task_id: str


@dataclass(frozen=True)
class Deadlock:
    """No task can make progress; ``chain`` names the stuck waits."""

    signature: str
    chain: list[str]


Action = Start | MakeReady | CancelTask | Finalize | Deadlock


def _outcome_line(task: Task) -> str:
    oc = task.outcome
    status = oc.status if oc else task.status
    label = {"done": "已完成", "failed": "失败", "cancelled": "已取消"}.get(status, status)
    line = f"- {task.id}「{task.title}」{label}"
    if oc is not None:
        if oc.summary:
            line += f"：{oc.summary.strip()[:1200]}"
        if oc.reason:
            line += f"（原因：{oc.reason}）"
        if oc.files_changed:
            line += f"\n  改动文件：{', '.join(oc.files_changed[:20])}"
    if task.session_file:
        line += f"\n  完整记录：{task.session_file}"
    return line


def _question_line(task: Task, qid: str) -> str:
    q = next((q for q in task.questions if q.id == qid), None)
    if q is None or q.answer is None:
        return ""
    return f"- 主 agent 回答了你的问题「{q.text[:120]}」：{q.answer}"


def dependency_note(board: TaskBoard, task: Task, ids: Iterable[str]) -> str:
    lines: list[str] = []
    for dep in ids:
        if dep.startswith(QUESTION_PREFIX):
            line = _question_line(task, dep[len(QUESTION_PREFIX) :])
        else:
            other = board.tasks.get(dep)
            line = _outcome_line(other) if other is not None else ""
        if line:
            lines.append(line)
    return "\n".join(lines)


def _settled(board: TaskBoard, task: Task, dep: str) -> bool:
    if dep.startswith(QUESTION_PREFIX):
        qid = dep[len(QUESTION_PREFIX) :]
        q = next((q for q in task.questions if q.id == qid), None)
        return q is None or q.answer is not None
    other = board.tasks.get(dep)
    return other is None or other.terminal


def _failed(board: TaskBoard, dep: str) -> bool:
    other = board.tasks.get(dep)
    return other is not None and other.status in ("failed", "cancelled")


def reconcile(board: TaskBoard, *, max_parallel: int) -> list[Action]:
    """Derive the next scheduling actions from the board (design §6.1).

    Pure: reads the board, returns actions; the supervisor executes them
    (each execution writes records, then reconcile runs again until it
    returns nothing new). Order: cascades, finalizes, settled waits,
    ready deps, dispatch, deadlock.
    """
    actions: list[Action] = []
    tasks = list(board.tasks.values())

    # 1. Owner cascade: live work below a cancelled/failed owner stops.
    for task in tasks:
        if task.terminal or task.owner is None:
            continue
        owner = board.tasks.get(task.owner)
        if owner is not None and owner.status in ("cancelled", "failed"):
            actions.append(CancelTask(task.id, f"所属任务 {owner.id} 已{owner.status}"))
    if actions:
        return actions

    # 2. completing → terminal once owned work drained.
    for task in tasks:
        if task.status == "completing" and all(t.terminal for t in board.owned_by(task.id)):
            actions.append(Finalize(task.id))

    # 3. Settled waits (need / ask_main) → ready, with the outcomes.
    for task in tasks:
        if task.status != "waiting":
            continue
        settled = [d for d in task.waiting_on if _settled(board, task, d)]
        failed_now = task.join == "fail_fast" and any(_failed(board, d) for d in settled)
        if len(settled) == len(task.waiting_on) or failed_now:
            if task.on_dep_fail == "cascade" and any(_failed(board, d) for d in settled):
                actions.append(CancelTask(task.id, "等待的依赖失败或被取消"))
            else:
                actions.append(MakeReady(task.id, dependency_note(board, task, settled)))

    # 4. pending → ready when depends_on are all terminal.
    for task in tasks:
        if task.status != "pending":
            continue
        if all(_settled(board, task, d) for d in task.depends_on):
            if task.on_dep_fail == "cascade" and any(_failed(board, d) for d in task.depends_on):
                actions.append(CancelTask(task.id, "前置依赖失败或被取消"))
            else:
                actions.append(MakeReady(task.id, dependency_note(board, task, task.depends_on)))
    if actions:
        return actions

    # 5. Dispatch ready tasks into free slots: user priority, then work
    # already in progress (it holds context), then the critical path,
    # then creation order.
    running = len(board.with_status(*ACTIVE))
    free = max_parallel - running
    if free > 0:
        ready = board.with_status("ready")
        ready.sort(
            key=lambda t: (
                -t.priority,
                0 if t.session_file else 1,
                -board.critical_path(t.id),
                int(t.id[1:]) if t.id[1:].isdigit() else 0,
            )
        )
        actions.extend(Start(t.id) for t in ready[:free])
    if actions:
        return actions

    # 6. Deadlock: live work exists but nothing can move by itself.
    # Paused and proposed work is waiting on the user, not stuck.
    progressing = board.with_status("running", "ready", "completing", "paused", "proposed")
    stuck = [
        t
        for t in board.with_status("waiting", "pending")
        if not any(d.startswith(QUESTION_PREFIX) for d in t.waiting_on)
    ]
    if stuck and not progressing:
        chain = [f"{t.id} 等待 {', '.join(t.waiting_on or t.depends_on)}" for t in stuck]
        signature = "|".join(sorted(chain))
        actions.append(Deadlock(signature, chain))
    return actions


def new_notice_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(3)}"
