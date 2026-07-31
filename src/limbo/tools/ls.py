"""List directory contents tool."""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, truncate_head

MAX_ENTRIES = 500


class LsTool(BaseTool):
    name = "ls"
    description = (
        "List directory contents. Includes dotfiles. Output is truncated to "
        f"{MAX_ENTRIES} entries (a notice is shown when the limit is hit)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list"},
            "limit": {"type": "integer", "description": "Max entries"},
        },
        "required": [],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_ENTRIES)

        target = self.resolve_existing(path, kind="dir")

        try:
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            return ToolResult(success=False, error=f"Could not list directory: {e}")

        lines = []
        for entry in entries[:limit]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")

        if not lines:
            return ToolResult(success=True, output="(empty directory)")

        output = "\n".join(lines)
        notices = []
        if len(entries) > limit:
            # Never slice silently: the model must know entries are missing.
            notices.append(
                f"Showing {limit} of {len(entries)} entries. "
                f"Use limit={limit * 2} for more."
            )
        truncation = truncate_head(output, max_lines=len(lines) + 1)
        if truncation.truncated:
            output = truncation.content
            notices.append("Output truncated by size limit.")
        if notices:
            output += f"\n\n[{' '.join(notices)}]"
        return ToolResult(success=True, output=output)
