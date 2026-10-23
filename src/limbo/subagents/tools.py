"""Task tools: what the main agent and the subagents can call (design §7.2).

All of them are async tools (``is_async``): they drive the live board on
the event loop. Replay policy: only ``task_status`` and ``checkpoint`` are
``safe``; re-running an interrupted ``cancel`` or ``finish`` after a
crash could undo newer work, so the model is told they may have run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from limbo.models import ToolResult
from limbo.subagents.board import BoardError
from limbo.tools.base import BaseTool

if TYPE_CHECKING:
    from limbo.subagents.supervisor import Supervisor


def _ok(text: str, *, end_turn: bool = False) -> ToolResult:
    return ToolResult(success=True, output=text, end_turn=end_turn)


def _err(text: str) -> ToolResult:
    return ToolResult(success=False, error=text)


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = [v.strip() for v in value.split(",")]
        value = parsed if isinstance(parsed, list) else [parsed]
    if not isinstance(value, list):
        return [str(value)]
    return [str(v).strip() for v in value if str(v).strip()]


class _TaskTool(BaseTool):
    is_async = True

    def __init__(self, workdir: Path, supervisor: Supervisor, task_id: str | None = None):
        super().__init__(workdir)
        self.sup = supervisor
        self.task_id = task_id

    def run(self, arguments: dict[str, Any]) -> ToolResult:  # pragma: no cover
        return _err(f"{self.name} must run on the event loop")

    async def aexecute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return await self.arun(arguments)
        except BoardError as e:
            return _err(str(e))


# -- main agent ---------------------------------------------------------------

_TASK_SPEC = {
    "type": "object",
    "properties": {
        "key": {
            "type": "string",
            "description": "Short local key other tasks in this plan use in depends_on",
        },
        "title": {"type": "string"},
        "brief": {
            "type": "string",
            "description": (
                "Everything the subagent needs: goal, context, constraints, where to look"
            ),
        },
        "profile": {
            "type": "string",
            "description": "Agent profile name from <available_agents> (default worker)",
        },
        "depends_on": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keys from this plan or existing task ids that must finish first",
        },
        "acceptance": {
            "type": "string",
            "description": "How the result is verified (a command or a criterion)",
        },
        "scope_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Glob paths this task will write (claims them)",
        },
        "priority": {"type": "integer"},
        "isolation": {"type": "string", "enum": ["shared", "worktree"]},
        "context": {
            "type": "string",
            "enum": ["brief", "fork"],
            "description": "fork = the subagent starts with a copy of this conversation",
        },
    },
    "required": ["title", "brief"],
}


class PlanTasksTool(_TaskTool):
    name = "plan_tasks"
    description = (
        "Delegate work to background subagents by submitting a task DAG. "
        "Use it when the work splits into independent parts, needs broad "
        "exploration whose details you do not need, or benefits from an "
        "independent review/verification. Subagents run in parallel while "
        "you keep talking to the user; their results come back to you as "
        "<subagent_updates> messages. Submit the whole plan in one call."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {"type": "array", "items": _TASK_SPEC},
            "rationale": {"type": "string", "description": "One sentence: why delegate"},
        },
        "required": ["tasks"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        specs = arguments.get("tasks")
        if isinstance(specs, str):
            try:
                specs = json.loads(specs)
            except ValueError:
                return _err("tasks must be a JSON array of task objects")
        if not isinstance(specs, list) or not specs:
            return _err("tasks must be a non-empty array")
        result = self.sup.plan([s for s in specs if isinstance(s, dict)], created_by="main")
        return _ok(result.message, end_turn=result.proposed)


class TaskStatusTool(_TaskTool):
    name = "task_status"
    replay = "safe"
    description = (
        "Show the task board: every task's state, last checkpoint and open "
        "questions; with an id, that task in detail (brief, checkpoints, "
        "outcome, transcript path)."
    )
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Task id or agent name"}},
        "required": [],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        return _ok(self.sup.status_text(arguments.get("id") or None))


class TaskSendTool(_TaskTool):
    name = "task_send"
    description = (
        "Send a message to a subagent. mode=steer reaches it at its next "
        "step; follow_up waits until its current turn ends. A message to a "
        "task that is not running is delivered when it (re)starts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Task id or agent name"},
            "text": {"type": "string"},
            "mode": {"type": "string", "enum": ["steer", "follow_up"]},
        },
        "required": ["id", "text"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        mode = "follow_up" if arguments.get("mode") == "follow_up" else "steer"
        return _ok(
            self.sup.send(str(arguments.get("id", "")), str(arguments.get("text", "")), mode=mode)
        )


class AnswerTool(_TaskTool):
    name = "answer"
    description = (
        "Answer a subagent's question (the id from its <question> update). "
        "It resumes with the answer."
    )
    parameters = {
        "type": "object",
        "properties": {"question_id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["question_id", "text"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        return _ok(
            self.sup.answer(str(arguments.get("question_id", "")), str(arguments.get("text", "")))
        )


class TaskControlTool(_TaskTool):
    name = "task_control"
    description = (
        "Control tasks on the user's behalf: pause, resume, cancel, retry "
        "(fresh=true starts a new transcript), reprioritize (priority), "
        "edit (brief/depends_on of a task that has not started, or a "
        "correction steered into a running one), approve / reject proposed "
        "tasks, merge (a finished worktree task's branch). id may be 'all' "
        "for pause/resume/cancel/approve."
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Task id, agent name, or 'all'"},
            "action": {
                "type": "string",
                "enum": [
                    "pause",
                    "resume",
                    "cancel",
                    "retry",
                    "reprioritize",
                    "edit",
                    "approve",
                    "reject",
                    "merge",
                ],
            },
            "note": {
                "type": "string",
                "description": "Reason, or what the subagent is told on resume/retry",
            },
            "priority": {"type": "integer"},
            "fresh": {"type": "boolean"},
            "brief": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "action"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        return _ok(
            self.sup.control(
                str(arguments.get("id", "")),
                str(arguments.get("action", "")),
                note=str(arguments.get("note") or ""),
                priority=arguments.get("priority"),
                fresh=bool(arguments.get("fresh")),
                brief=arguments.get("brief"),
                depends_on=_list(arguments["depends_on"]) if "depends_on" in arguments else None,
                by="main",
            )
        )


def main_tools(workdir: Path, sup: Supervisor) -> list[BaseTool]:
    return [
        PlanTasksTool(workdir, sup),
        TaskStatusTool(workdir, sup),
        TaskSendTool(workdir, sup),
        AnswerTool(workdir, sup),
        TaskControlTool(workdir, sup),
    ]


# -- subagent -----------------------------------------------------------------


class CheckpointTool(_TaskTool):
    name = "checkpoint"
    replay = "safe"
    description = (
        "Record progress on your task at each phase change (what is done, "
        "what is next). The main agent and the user see it on the task board."
    )
    parameters = {
        "type": "object",
        "properties": {
            "phase": {"type": "string", "description": "Short phase name"},
            "summary": {"type": "string"},
            "files": {"type": "array", "items": {"type": "string"}},
            "progress": {"type": "number", "description": "0..1"},
        },
        "required": ["phase"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        assert self.task_id is not None
        progress = arguments.get("progress")
        try:
            value = float(progress) if progress is not None else None
        except (TypeError, ValueError):
            value = None
        self.sup.child_checkpoint(
            self.task_id,
            str(arguments.get("phase", "")),
            str(arguments.get("summary") or ""),
            _list(arguments.get("files")),
            value,
        )
        return _ok("已打点")


class NeedTool(_TaskTool):
    name = "need"
    description = (
        "You discovered that your task depends on other work. on=[task ids] "
        "waits for existing tasks; new={title, brief, profile} requests work "
        "that does not exist yet. policy=wait suspends you until it is done "
        "(you resume with its result); cancel_self drops your task as "
        "obsolete; ask_main asks the main agent to decide. Ends your turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "on": {"type": "array", "items": {"type": "string"}},
            "new": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "brief": {"type": "string"},
                    "profile": {"type": "string"},
                    "scope_paths": {"type": "array", "items": {"type": "string"}},
                },
            },
            "reason": {"type": "string"},
            "policy": {"type": "string", "enum": ["wait", "cancel_self", "ask_main"]},
            "join": {"type": "string", "enum": ["all_settled", "fail_fast"]},
        },
        "required": ["reason"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        assert self.task_id is not None
        new = arguments.get("new")
        if isinstance(new, str):
            try:
                new = json.loads(new)
            except ValueError:
                new = {"title": new, "brief": new}
        message = self.sup.child_need(
            self.task_id,
            on=_list(arguments.get("on")),
            new=new if isinstance(new, dict) else None,
            reason=str(arguments.get("reason") or ""),
            policy=str(arguments.get("policy") or "wait"),
            join=str(arguments.get("join") or "all_settled"),
        )
        return _ok(message, end_turn=True)


class AskMainTool(_TaskTool):
    name = "ask_main"
    description = (
        "Ask the main agent (and through it the user) a question you cannot "
        "decide yourself. You are suspended until it is answered. Ends your turn."
    )
    parameters = {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        assert self.task_id is not None
        return _ok(
            self.sup.child_ask(self.task_id, str(arguments.get("question", ""))), end_turn=True
        )


class FinishTool(_TaskTool):
    name = "finish"
    description = (
        "Deliver your task and stop. summary: what you did and how you "
        "verified it (the main agent reads only this). status=failed when "
        "you could not complete it (say why). Ends your turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "files_changed": {"type": "array", "items": {"type": "string"}},
            "follow_ups": {"type": "array", "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["done", "failed"]},
        },
        "required": ["summary"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        assert self.task_id is not None
        message = self.sup.child_finish(
            self.task_id,
            summary=str(arguments.get("summary") or ""),
            files=_list(arguments.get("files_changed")),
            follow_ups=_list(arguments.get("follow_ups")),
            failed=arguments.get("status") == "failed",
        )
        return _ok(message, end_turn=True)


class ProposeTaskTool(_TaskTool):
    name = "propose_task"
    description = (
        "Propose a new task you noticed but that is outside your scope "
        "(you cannot start subagents yourself). The main agent decides. "
        "Does not suspend you."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "brief": {"type": "string"},
            "profile": {"type": "string"},
        },
        "required": ["title", "brief"],
    }

    async def arun(self, arguments: dict[str, Any]) -> ToolResult:
        assert self.task_id is not None
        return _ok(
            self.sup.child_propose(
                self.task_id,
                str(arguments.get("title", "")),
                str(arguments.get("brief", "")),
                str(arguments.get("profile") or "worker"),
            )
        )


def subagent_tools(workdir: Path, sup: Supervisor, task_id: str) -> list[BaseTool]:
    return [
        CheckpointTool(workdir, sup, task_id),
        NeedTool(workdir, sup, task_id),
        AskMainTool(workdir, sup, task_id),
        FinishTool(workdir, sup, task_id),
        ProposeTaskTool(workdir, sup, task_id),
    ]
