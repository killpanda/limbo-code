"""Surgical file edit tool."""

from __future__ import annotations

import difflib
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir


class EditTool(BaseTool):
    name = "edit"
    description = (
        "Make surgical edits to a file by replacing exact text. old_text must match exactly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "old_text": {
                "type": "string",
                "description": "Exact text to find and replace",
            },
            "new_text": {
                "type": "string",
                "description": "New text to replace with",
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        raw_path = arguments.get("path", "")
        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")
        target = (self.workdir / raw_path).resolve()
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")
        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {raw_path}")

        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            return ToolResult(success=False, error=f"old_text not found in {raw_path}.")
        if occurrences > 1:
            return ToolResult(success=False, error=f"Text must be unique in {raw_path}.")

        new_content = content.replace(old_text, new_text, 1)
        if new_content == content:
            return ToolResult(success=False, error=f"No changes to {raw_path}.")

        diff = self._make_diff(content, new_content)

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Proposed edit to {raw_path}:\n{diff}",
                requires_confirmation=True,
            )

        try:
            target.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not write file: {e}")

        return ToolResult(success=True, output=f"Edited {raw_path}.\n{diff}")

    def _make_diff(self, old: str, new: str) -> str:
        return "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
