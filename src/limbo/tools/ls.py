"""List directory contents tool."""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, resolve_path

MAX_ENTRIES = 500


class LsTool(BaseTool):
    name = "ls"
    description = "List directory contents. Includes dotfiles."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list"},
            "limit": {"type": "integer", "description": "Max entries"},
        },
        "required": [],
    }

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_ENTRIES)

        target = resolve_path(path, self.workdir)
        if isinstance(target, ToolResult):
            return target
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        if not target.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}")

        try:
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            return ToolResult(success=False, error=f"Could not list directory: {e}")

        lines = []
        for entry in entries[:limit]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")

        return ToolResult(success=True, output="\n".join(lines))
