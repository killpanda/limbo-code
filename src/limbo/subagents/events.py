"""Background events the supervisor publishes to frontends, and notices.

Frontends subscribe through ``Conversation.subscribe``; every event is
delivered on the event loop. Non-UI layer: never imports textual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from limbo.subagents.board import Task

# result: a task finished (delivered to the main agent; batched)
# question: a subagent asks / a proposal waits for approval (wakes main)
# alert: failure, deadlock, budget (wakes main)
# info: the user acted directly (/task, @name); delivered, never wakes
# progress: checkpoints and state changes — UI only, never delivered
NoticeKind = Literal["result", "question", "alert", "info", "progress"]
DELIVERED_KINDS: frozenset[str] = frozenset({"result", "question", "alert", "info"})


@dataclass(frozen=True)
class Notice:
    id: str
    task_id: str
    kind: NoticeKind
    text: str
    # For the UI: one line, no XML.
    headline: str = ""


@dataclass(frozen=True)
class TaskChanged:
    """A task's state, checkpoint or activity line changed."""

    task: Task
    activity: str = ""


@dataclass(frozen=True)
class NoticeEvent:
    notice: Notice


@dataclass(frozen=True)
class MainWake:
    """Notices are queued for an idle main agent: start a turn for them
    (``Conversation.take_wake_input``)."""


@dataclass(frozen=True)
class PlanProposed:
    """``plan_tasks`` in suggest mode: these tasks wait for the user."""

    task_ids: list[str] = field(default_factory=list)


BackgroundEvent = TaskChanged | NoticeEvent | MainWake | PlanProposed
