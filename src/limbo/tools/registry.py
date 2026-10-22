"""Tool registry that collects all available tools."""

from __future__ import annotations

import asyncio
import contextvars
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from limbo.config import Config
from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ReplayPolicy
from limbo.tools.bash import BashTool
from limbo.tools.edit import EditTool
from limbo.tools.find import FindTool
from limbo.tools.grep import GrepTool
from limbo.tools.hooks import Block, PlanModeHook, Rewrite, ToolHook
from limbo.tools.ls import LsTool
from limbo.tools.read import ReadTool
from limbo.tools.run_code import RunCodeTool
from limbo.tools.write import WriteTool

# The tool call currently executing in this context. asyncio.to_thread and
# run_code's private event loop both copy the context, so a Code Mode
# sub-call sees its run_code call here (its parent).
_CURRENT_CALL: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "limbo_current_tool_call", default=None
)


class ToolRegistry:
    """Registers and executes tools."""

    def __init__(self, workdir: Path, config: Config | None = None):
        self.workdir = workdir
        self.config = config or Config()
        # Code Mode (RFC parity with deepseek-harness): when True, the
        # model-facing catalog collapses to the single `run_code` transport
        # and every other tool is reached from inside a program (see
        # limbo.code_mode). Switched by /code-mode; the agent rebuilds the
        # system prompt on toggle.
        self.code_mode = False
        self._tools: dict[str, BaseTool] = {}
        # Trace sink for Code Mode sub-calls (set by the agent): each one is
        # logged as tool_call/tool_result with its run_code parent_call_id.
        self.trace_log: Callable[..., None] | None = None
        # before_tool/after_tool chain (tools/hooks.py). Empty by default:
        # no safety fences; /plan mounts the read-only PlanModeHook.
        self.hooks: list[ToolHook] = []
        for tool_class in [EditTool, WriteTool, LsTool, ReadTool, GrepTool, FindTool]:
            self.register(tool_class)  # type: ignore[type-abstract]
        if self.config.tools.bash_enabled:
            self._tools["bash"] = BashTool(workdir=workdir)
        # The run_code transport is always registered — under code mode it
        # is the ONLY model-facing tool; under native mode it is hidden
        # from definitions() so nothing changes.
        run_code = RunCodeTool(workdir=workdir)
        run_code.registry = self
        self._tools["run_code"] = run_code

    def register(self, tool_class: type[BaseTool]) -> None:
        tool = tool_class(workdir=self.workdir)
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def replay_policy(self, name: str) -> ReplayPolicy:
        """How a call to ``name`` is recovered after a crash (unknown: unsafe)."""
        policy = getattr(self.get(name), "replay", "unsafe")
        return "safe" if policy == "safe" else "unsafe"

    async def execute(
        self, name: str, arguments: dict[str, Any], call_id: str | None = None
    ) -> ToolResult:
        parent = _CURRENT_CALL.get()
        if parent is not None and call_id is None:
            # A Code Mode sub-call: give it an id and trace it under its
            # run_code parent (sub-calls never enter history).
            call_id = f"{parent}.{secrets.token_hex(3)}"
            return await self._execute_traced(name, arguments, call_id, parent)
        return await self._execute(name, arguments, call_id)

    async def _execute(
        self, name: str, arguments: dict[str, Any], call_id: str | None
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        for hook in list(self.hooks):
            try:
                decision = hook.before_tool(name, arguments)
            except Exception as e:  # noqa: BLE001 - a broken hook fails closed
                return ToolResult(
                    success=False, error=f"Blocked by hook {hook.name}: {e}"
                )
            if isinstance(decision, Block):
                return ToolResult(success=False, error=decision.reason)
            if isinstance(decision, Rewrite):
                arguments = decision.arguments
        token = _CURRENT_CALL.set(call_id) if call_id is not None else None
        try:
            result = await asyncio.to_thread(tool.execute, arguments)
        finally:
            if token is not None:
                _CURRENT_CALL.reset(token)
        for hook in list(self.hooks):
            try:
                replaced = hook.after_tool(name, arguments, result)
            except Exception as e:  # noqa: BLE001
                return ToolResult(
                    success=False,
                    error=f"Hook {hook.name} failed after {name} ran: {e}",
                )
            if replaced is not None:
                result = replaced
        return result

    async def _execute_traced(
        self, name: str, arguments: dict[str, Any], call_id: str, parent: str
    ) -> ToolResult:
        log = self.trace_log
        if log is not None:
            log("tool_call", id=call_id, parent_call_id=parent, name=name,
                arguments=arguments)
        start = time.monotonic()
        result = await self._execute(name, arguments, call_id)
        if log is not None:
            log(
                "tool_result",
                id=call_id,
                parent_call_id=parent,
                name=name,
                success=result.success,
                output=result.output,
                error=result.error,
                duration=time.monotonic() - start,
            )
        return result

    # -- plan mode (read-only hook) ----------------------------------------

    @property
    def plan_mode(self) -> bool:
        return any(isinstance(h, PlanModeHook) for h in self.hooks)

    @plan_mode.setter
    def plan_mode(self, enabled: bool) -> None:
        if enabled == self.plan_mode:
            return
        if enabled:
            self.hooks.append(
                PlanModeHook(lambda n: self.replay_policy(n) == "safe")
            )
        else:
            self.hooks = [h for h in self.hooks if not isinstance(h, PlanModeHook)]

    def cancel_active(self) -> None:
        """Cancel in-flight interruptible tools (ESC interrupt, RFC LIM-53).

        Only bash is interruptible (process-tree kill); file tools are
        no-ops by design — they run to completion and record real results.
        """
        for tool in self._tools.values():
            tool.cancel()

    def definitions(self) -> list[dict[str, Any]]:
        """The model-facing tool definitions.

        Under code mode this collapses to ``[run_code]`` only — the model
        reaches every other tool from inside a program (deepseek-harness
        Code Mode parity).
        """
        names = ["run_code"] if self.code_mode else [
            n for n in self._tools if n != "run_code"
        ]
        return [self._definition(self._tools[name]) for name in names]

    def all_definitions(self) -> list[dict[str, Any]]:
        """Every tool definition except the run_code transport.

        The SDK section is rendered from this view so the model never sees
        ``run_code`` listed as a program-callable method (no nested runs).
        """
        return [
            self._definition(tool)
            for name, tool in self._tools.items()
            if name != "run_code"
        ]

    @staticmethod
    def _definition(tool: BaseTool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
