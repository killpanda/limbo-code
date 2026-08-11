"""Search file contents tool."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import (
    BaseTool,
    truncate_output,
)

MAX_MATCHES = 100


class GrepTool(BaseTool):
    name = "grep"
    description = (
        "Search file contents for a pattern. Requires ripgrep (rg) on PATH. "
        "Respects .gitignore, glob filters, and context lines."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern"},
            "path": {"type": "string", "description": "Directory or file to search"},
            "glob": {
                "type": "string",
                "description": "Filter files by glob (e.g. '*.py', 'src/**/*.ts')",
            },
            "ignore_case": {"type": "boolean", "description": "Case-insensitive"},
            "fixed_string": {"type": "boolean", "description": "Literal string"},
            "context": {"type": "integer", "description": "Lines before/after match"},
            "limit": {
                "type": "integer",
                "description": "Max matches per file",
            },
        },
        "required": ["pattern"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        glob = arguments.get("glob")
        ignore_case = arguments.get("ignore_case", False)
        fixed_string = arguments.get("fixed_string", False)
        context = arguments.get("context", 0)
        limit = arguments.get("limit", MAX_MATCHES)

        target = self.resolve_existing(path)

        # ripgrep is a hard dependency (pi parity): a single implementation
        # means one set of gitignore/glob/context semantics, no drift
        # between a full-featured path and a degraded Python fallback.
        rg = shutil.which("rg")
        if rg is None:
            return ToolResult(
                success=False,
                error=(
                    "ripgrep (rg) is required for grep but was not found on "
                    "PATH. Install it (e.g. 'brew install ripgrep' or "
                    "'apt install ripgrep') and try again."
                ),
            )
        return self._run_rg(
            rg, target, pattern, glob, ignore_case, fixed_string, context, limit
        )

    def _run_rg(
        self,
        rg: str,
        target: Path,
        pattern: str,
        glob: str | None,
        ignore_case: bool,
        fixed_string: bool,
        context: int,
        limit: int,
    ) -> ToolResult:
        cmd = [rg, "--line-number", "--no-heading", "--color=never", "--no-require-git"]
        if ignore_case:
            cmd.append("--ignore-case")
        if fixed_string:
            cmd.append("--fixed-strings")
        if context:
            cmd.extend(["-C", str(context)])
        if glob:
            cmd.extend(["-g", glob])
        # Emit workdir-relative paths when possible; targets outside the
        # workdir are passed as absolute paths (rg accepts both).
        try:
            rel_target = target.relative_to(self.workdir)
        except ValueError:
            rel_target = target
        cmd.extend(["--max-count", str(limit), "--", pattern, str(rel_target)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workdir),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="grep timed out.")
        except FileNotFoundError:
            return ToolResult(success=False, error="ripgrep not found.")

        output = proc.stdout
        output = truncate_output(output)

        return ToolResult(success=True, output=output or "No matches.")
