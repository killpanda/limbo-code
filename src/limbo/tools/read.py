"""Read file contents tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import Attachment, ToolResult
from limbo.tools.base import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    BaseTool,
    ToolError,
    truncate_head,
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB (text)
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB (images, pre-base64)

_SNIFF_BYTES = 4100


def detect_image_mime(path: Path) -> str | None:
    """Sniff the leading bytes for a supported image type (pi parity).

    Returns the MIME type for jpg/png/gif/webp/bmp, else None. Magic bytes
    are used instead of the extension so a mis-named file still behaves
    correctly.
    """
    try:
        with path.open("rb") as f:
            head = f.read(_SNIFF_BYTES)
    except OSError:
        return None
    if head.startswith(b"\xff\xd8\xff"):
        # JPEG; 0xF7 as the fourth byte marks a JPEG XL codestream.
        return None if len(head) > 3 and head[3] == 0xF7 else "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:3] == b"GIF":
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:2] == b"BM":
        return "image/bmp"
    return None


class ReadTool(BaseTool):
    name = "read"
    description = (
        "Read the contents of a file. Supports text files and images "
        "(jpg, png, gif, webp, bmp). Images are sent as attachments. "
        "For text files, output is truncated to the first "
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

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        target = self.resolve(raw_path)

        if not target.exists():
            raise ToolError(f"File not found: {raw_path}")
        if not target.is_file():
            raise ToolError(f"Not a file: {raw_path}")

        try:
            file_size = target.stat().st_size
        except OSError as e:
            return ToolResult(success=False, error=f"Could not read file: {e}")

        mime = detect_image_mime(target)
        if mime is not None:
            if file_size > MAX_IMAGE_SIZE:
                return ToolResult(
                    success=False,
                    error=(
                        f"Image too large ({file_size} bytes). "
                        f"Maximum size is {MAX_IMAGE_SIZE} bytes."
                    ),
                )
            return ToolResult(
                success=True,
                output=f"Read image file [{mime}]",
                attachments=[
                    Attachment(
                        kind="image",
                        name=target.name,
                        path=str(target),
                        mime=mime,
                    )
                ],
            )

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
        total_file_lines = len(lines)
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
            # Report absolute file line numbers: the truncation window is a
            # slice of the file starting at `offset` (default 1).
            first_line = offset or 1
            last_line = first_line + truncation.output_lines - 1
            output += (
                f"\n[Output truncated: showing lines {first_line}-"
                f"{last_line} of {total_file_lines}. "
                f"Use offset/limit to read more.]"
            )

        return ToolResult(success=True, output=output)
