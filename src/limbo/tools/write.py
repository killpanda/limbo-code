"""Write file contents tool."""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir


class WriteTool(BaseTool):
    name = "write"
    description = (
        "Create or overwrite a file. Use only for new files or complete rewrites."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        content = arguments.get("content", "")

        target = (self.workdir / raw_path).resolve()
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not write file: {e}")

        return ToolResult(
            success=True,
            output=f"File ready to write: {raw_path}",
            requires_confirmation=True,
        )
