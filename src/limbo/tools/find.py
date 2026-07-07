"""Find files by glob pattern tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool


MAX_RESULTS = 1000
MAX_BYTES = 512 * 1024


class FindTool(BaseTool):
    name = "find"
    description = "Find files by glob pattern. Respects .gitignore."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["pattern"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_RESULTS)

        target = (self.workdir / path).resolve()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            matches = sorted(target.glob(pattern))
        except ValueError as e:
            return ToolResult(success=False, error=f"Invalid glob pattern: {e}")

        results = []
        for p in matches:
            if not p.is_file():
                continue
            results.append(str(p.relative_to(self.workdir)))
            if len(results) >= limit:
                break

        output = "\n".join(results)
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"

        return ToolResult(success=True, output=output or "No matches.")
