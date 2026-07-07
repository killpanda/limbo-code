"""Read file contents tool."""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir

MAX_LINES = 2000
MAX_BYTES = 512 * 1024
SENSITIVE_FILES = {".env", "id_rsa", "id_ed25519", ".ssh"}


class ReadTool(BaseTool):
    name = "read"
    description = "Read the contents of a file. Use offset/limit for large files."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "offset": {
                "type": "integer",
                "description": "Line number to start from (1-indexed, optional)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (optional)",
            },
        },
        "required": ["path"],
    }

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        raw_path = arguments.get("path", "")
        target = self.workdir / raw_path
        target = target.resolve()

        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        if target.name in SENSITIVE_FILES or any(
            part in SENSITIVE_FILES for part in target.parts
        ):
            return ToolResult(success=False, error="Refusing to read sensitive file.")

        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {raw_path}")

        if not target.is_file():
            return ToolResult(success=False, error=f"Not a file: {raw_path}")

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not read file: {e}")

        lines = text.splitlines(keepends=True)
        offset = arguments.get("offset")
        limit = arguments.get("limit")

        if offset is not None:
            start = max(0, offset - 1)
            lines = lines[start:]
        if limit is not None:
            lines = lines[:limit]

        output = "".join(lines)
        truncated = False
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES]
            truncated = True
        elif len(lines) > MAX_LINES:
            output = "".join(lines[:MAX_LINES])
            truncated = True

        if truncated:
            output += "\n[Output truncated. Use offset/limit to read more.]"

        return ToolResult(success=True, output=output)
