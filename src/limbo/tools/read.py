"""Read file contents tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.config import DEFAULT_SENSITIVE_FILES
from limbo.models import ToolResult
from limbo.tools.base import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    BaseTool,
    ToolError,
    truncate_head,
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class ReadTool(BaseTool):
    name = "read"
    description = (
        "Read the contents of a file. Output is truncated to the first "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
        "(whichever is hit first). Use offset/limit for large files. "
        "When you need the full file, continue with offset until complete."
    )
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

    def __init__(
        self, workdir: Path, sensitive_files: list[str] | None = None
    ):
        super().__init__(workdir)
        self.sensitive_files = set(sensitive_files or DEFAULT_SENSITIVE_FILES)

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        target = self.resolve(raw_path)

        if target.name in self.sensitive_files or any(
            part in self.sensitive_files for part in target.parts
        ):
            return ToolResult(success=False, error="Refusing to read sensitive file.")

        if not target.exists():
            raise ToolError(f"File not found: {raw_path}")
        if not target.is_file():
            raise ToolError(f"Not a file: {raw_path}")

        try:
            file_size = target.stat().st_size
        except OSError as e:
            return ToolResult(success=False, error=f"Could not read file: {e}")
        if file_size > MAX_FILE_SIZE:
            return ToolResult(
                success=False,
                error=(
                    f"File too large ({file_size} bytes). "
                    f"Maximum size is {MAX_FILE_SIZE} bytes."
                ),
            )

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not read file: {e}")

        lines = text.splitlines(keepends=True)
        offset = arguments.get("offset")
        limit = arguments.get("limit")

        if offset is not None and offset < 1:
            return ToolResult(
                success=False, error="offset must be a positive integer."
            )
        if limit is not None and limit < 1:
            return ToolResult(
                success=False, error="limit must be a positive integer."
            )

        if offset is not None:
            start = max(0, offset - 1)
            lines = lines[start:]
        if limit is not None:
            lines = lines[:limit]

        selected = "".join(lines)
        truncation = truncate_head(selected)
        output = truncation.content
        if truncation.truncated:
            output += (
                f"\n[Output truncated: showing lines 1-"
                f"{truncation.output_lines} of {truncation.total_lines}. "
                f"Use offset/limit to read more.]"
            )

        return ToolResult(success=True, output=output)
