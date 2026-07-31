"""Search file contents tool."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from limbo.config import DEFAULT_SENSITIVE_FILES
from limbo.models import ToolResult
from limbo.tools.base import (
    BaseTool,
    is_sensitive_path,
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

    def __init__(
        self,
        workdir: Path,
        sensitive_files: list[str] | None = None,
        allowed_roots: set[Path] | None = None,
    ):
        super().__init__(workdir, allowed_roots=allowed_roots)
        # Convenience guardrail (aligned with read/find): sensitive files
        # are skipped so secrets don't leak into the model context by
        # accident. Not a security boundary — bash is not covered.
        self.sensitive_files = set(sensitive_files or DEFAULT_SENSITIVE_FILES)

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        glob = arguments.get("glob")
        ignore_case = arguments.get("ignore_case", False)
        fixed_string = arguments.get("fixed_string", False)
        context = arguments.get("context", 0)
        limit = arguments.get("limit", MAX_MATCHES)

        target = self.resolve_existing(path)

        if is_sensitive_path(target, self.sensitive_files):
            return ToolResult(
                success=False,
                error=(
                    "Refusing to search sensitive path. This is a convenience "
                    "guardrail against accidental secret exposure, not a "
                    "security boundary. Ask the user if access is genuinely "
                    "needed."
                ),
            )

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
        # Exclude sensitive files (same guardrail as read). ``**/name``
        # matches at any depth including the search root; for directory
        # names (e.g. .ssh) rg skips the whole subtree.
        for name in sorted(self.sensitive_files):
            cmd.extend(["-g", f"!**/{name}"])
        # Emit workdir-relative paths, consistent with find. A granted root
        # outside the workdir is passed as an absolute path (rg accepts both).
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
