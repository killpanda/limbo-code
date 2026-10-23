"""Tool hooks a subagent's registry gets from its profile (design §6.5).

- :class:`ReadOnlyHook` — ``tools: read_only`` profiles: plan mode's
  read-only rules, but the subagent's own task tools stay callable.
- :class:`NoEditHook` — ``tools: no_edit`` (verifier): everything except
  ``edit``/``write``.
- :class:`ClaimHook` — writer subagents on a shared workdir: ``edit`` and
  ``write`` refuse a path another live task claimed in ``scope_paths``.
  ``bash`` cannot be checked reliably and is left alone.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from pathlib import Path
from typing import Any

from limbo.subagents.board import TaskBoard
from limbo.tools.hooks import BeforeDecision, Block, PlanModeHook, ToolHook

# Task tools every subagent may call whatever its profile.
SUBAGENT_TOOL_NAMES = frozenset({"checkpoint", "need", "ask_main", "finish", "propose_task"})


class ReadOnlyHook(ToolHook):
    name = "subagent-read-only"

    def __init__(self, is_read_only_tool: Callable[[str], bool]) -> None:
        self._plan = PlanModeHook(is_read_only_tool)

    def before_tool(self, name: str, arguments: dict[str, Any]) -> BeforeDecision:
        if name in SUBAGENT_TOOL_NAMES:
            return None
        decision = self._plan.before_tool(name, arguments)
        if isinstance(decision, Block):
            return Block(
                f"This subagent is read-only: {name} is blocked. Report what "
                "should change in your finish() summary instead."
            )
        return decision


class NoEditHook(ToolHook):
    name = "subagent-no-edit"

    def before_tool(self, name: str, arguments: dict[str, Any]) -> BeforeDecision:
        if name in ("edit", "write", "run_code"):
            return Block(
                f"This subagent does not edit files: {name} is blocked. Report "
                "the fix in your finish() summary instead."
            )
        return None


def matches(rel_path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        pattern = pattern.strip().lstrip("./")
        if not pattern:
            continue
        if fnmatch.fnmatch(rel_path, pattern):
            return True
        # "dir/**" also covers "dir" itself and anything below it.
        base = pattern.removesuffix("/**").removesuffix("/*")
        if base != pattern and (rel_path == base or rel_path.startswith(base + "/")):
            return True
    return False


class ClaimHook(ToolHook):
    name = "subagent-claims"

    def __init__(self, task_id: str, board: TaskBoard, workdir: Path) -> None:
        self._task_id = task_id
        self._board = board
        self._workdir = workdir.resolve()

    def before_tool(self, name: str, arguments: dict[str, Any]) -> BeforeDecision:
        if name not in ("edit", "write"):
            return None
        raw = str(arguments.get("path") or "")
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self._workdir / path
        try:
            rel = path.resolve().relative_to(self._workdir).as_posix()
        except ValueError:
            return None
        for task in self._board.live():
            if task.id == self._task_id or task.isolation == "worktree":
                continue
            if task.scope_paths and matches(rel, task.scope_paths):
                return Block(
                    f"{rel} 由任务 {task.id}（{task.agent_name}：{task.title}）认领。"
                    f'不要直接改它：用 need(on=["{task.id}"]) 等它完成，'
                    "或用 ask_main 说明你为什么需要改。"
                )
        return None
