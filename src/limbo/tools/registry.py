"""Tool registry that collects all available tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from limbo.config import Config
from limbo.models import ToolResult
from limbo.tools.base import BaseTool
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
        self._tools: dict[str, BaseTool] = {}
        for tool_class in [EditTool, WriteTool, LsTool, ReadTool, GrepTool, FindTool]:
            self.register(tool_class)  # type: ignore[type-abstract]
        if self.config.tools.bash_enabled:
            self._tools["bash"] = BashTool(workdir=workdir)

    def register(self, tool_class: type[BaseTool]) -> None:
        tool = tool_class(workdir=self.workdir)
        self._tools[tool.name] = tool

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
