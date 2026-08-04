"""Tool registry that collects all available tools."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from limbo.config import Config
from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir
from limbo.tools.bash import BashTool
from limbo.tools.edit import EditTool
from limbo.tools.find import FindTool
from limbo.tools.grep import GrepTool
from limbo.tools.ls import LsTool
from limbo.tools.read import ReadTool
from limbo.tools.write import WriteTool


class ToolRegistry:
    """Registers and executes tools."""

    def __init__(self, workdir: Path, config: Config | None = None):
        self.workdir = workdir
        self.config = config or Config()
        # Session-scoped roots granted beyond the workdir (implicit user
        # grants). One shared set, passed by reference to every tool, so a
        # grant mid-session takes effect on the next tool call.
        self.allowed_roots: set[Path] = set()
        self._tools: dict[str, BaseTool] = {}
        # Tools without configurable safety settings are registered generically.
        for tool_class in [EditTool, WriteTool, LsTool]:
            self.register(tool_class)  # type: ignore[type-abstract]
        # Wire configurable safety options from Config. The sensitive-file
        # guardrail is shared by read/grep/find so it behaves identically
        # across file tools.
        sensitive_files = self.config.safety.sensitive_files
        self._tools["read"] = ReadTool(
            workdir=workdir,
            sensitive_files=sensitive_files,
            allowed_roots=self.allowed_roots,
        )
        self._tools["grep"] = GrepTool(
            workdir=workdir,
            sensitive_files=sensitive_files,
            allowed_roots=self.allowed_roots,
        )
        self._tools["find"] = FindTool(
            workdir=workdir,
            sensitive_files=sensitive_files,
            allowed_roots=self.allowed_roots,
        )
        if self.config.tools.bash_enabled:
            self._tools["bash"] = BashTool(
                workdir=workdir,
                dangerous_patterns=self.config.safety.dangerous_commands,
            )

    def register(self, tool_class: type[BaseTool]) -> None:
        tool = tool_class(workdir=self.workdir, allowed_roots=self.allowed_roots)
        self._tools[tool.name] = tool

    def add_allowed_roots(self, paths: Iterable[Path]) -> list[Path]:
        """Grant session-scoped access to extra roots; returns newly added.

        Callers pre-filter candidates (existence, dir-vs-file semantics).
        Paths already inside the workdir or an existing grant are skipped.
        """
        added: list[Path] = []
        try:
            home = Path.home().resolve()
        except (OSError, RuntimeError):
            home = None
        for raw in sorted({str(p) for p in paths}):
            try:
                root = Path(raw).expanduser().resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            # Fence-voiding grants are never allowed, whatever the caller.
            if root == root.parent or (home is not None and root == home):
                continue
            if is_within_workdir(root, self.workdir):
                continue
            if any(is_within_workdir(root, g) for g in self.allowed_roots):
                continue
            self.allowed_roots.add(root)
            added.append(root)
        return added

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    async def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return await asyncio.to_thread(tool.execute, arguments)

    def cancel_active(self) -> None:
        """Cancel in-flight interruptible tools (ESC interrupt, RFC LIM-53).

        Only bash is interruptible (process-tree kill); file tools are
        no-ops by design — they run to completion and record real results.
        """
        for tool in self._tools.values():
            tool.cancel()

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
