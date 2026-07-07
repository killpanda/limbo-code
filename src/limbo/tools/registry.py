"""Tool registry that collects all available tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self._tools: dict[str, BaseTool] = {}
        for tool_class in [ReadTool, BashTool, EditTool, WriteTool, GrepTool, FindTool, LsTool]:
            self.register(tool_class)  # type: ignore[arg-type]

    def register(self, tool_class: type[BaseTool]) -> None:
        tool = tool_class(workdir=self.workdir)
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(arguments)

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
