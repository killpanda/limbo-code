"""Write file contents tool."""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, resolve_path


class WriteTool(BaseTool):
    name = "write"
    description = "Create or overwrite a file. Use only for new files or complete rewrites."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        raw_path = arguments.get("path", "")
        content = arguments.get("content", "")
        target = resolve_path(raw_path, self.workdir, strict=False)
        if isinstance(target, ToolResult):
            return target

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Will create/overwrite {raw_path} ({len(content)} chars).",
                requires_confirmation=True,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not write file: {e}")

        return ToolResult(success=True, output=f"Wrote {raw_path}.")
