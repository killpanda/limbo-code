"""Search file contents tool."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir
from limbo.tools.find import _GitignoreMatcher

MAX_MATCHES = 100
MAX_BYTES = 512 * 1024


class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents for a pattern. Respects .gitignore."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern"},
            "path": {"type": "string", "description": "Directory or file to search"},
            "glob": {"type": "string", "description": "Filter files by glob"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive"},
            "fixed_string": {"type": "boolean", "description": "Literal string"},
            "context": {"type": "integer", "description": "Lines before/after match"},
            "limit": {"type": "integer", "description": "Max matches"},
        },
        "required": ["pattern"],
    }

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        glob = arguments.get("glob")
        ignore_case = arguments.get("ignore_case", False)
        fixed_string = arguments.get("fixed_string", False)
        context = arguments.get("context", 0)
        limit = arguments.get("limit", MAX_MATCHES)

        target = (self.workdir / path).resolve()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        rg = self._find_rg()
        if rg:
            return self._run_rg(
                rg, target, pattern, glob, ignore_case, fixed_string, context, limit
            )
        return self._run_python_regex(target, pattern, ignore_case, fixed_string, limit)

    def _find_rg(self) -> str | None:
        import shutil

        return shutil.which("rg")

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
        cmd.extend(["--max-count", str(limit), pattern, str(target)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="grep timed out.")
        except FileNotFoundError:
            return ToolResult(success=False, error="ripgrep not found.")

        output = proc.stdout
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"

        return ToolResult(success=True, output=output or "No matches.")

    def _run_python_regex(
        self,
        target: Path,
        pattern: str,
        ignore_case: bool,
        fixed_string: bool,
        limit: int,
    ) -> ToolResult:
        # Fallback path: supports pattern/ignore_case/fixed_string/limit. The
        # ``context`` and ``glob`` parameters are not implemented in this path;
        # install ripgrep for full parameter support.
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(
                re.escape(pattern) if fixed_string else pattern, flags
            )
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        matches = []
        matcher = _GitignoreMatcher(self.workdir)
        files = [target] if target.is_file() else target.rglob("*")
        count = 0
        for f in files:
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(self.workdir)
            except ValueError:
                continue
            if matcher.is_ignored(str(rel)):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches.append(f"{rel}:{lineno}:{line}")
                    count += 1
                    if count >= limit:
                        break
            if count >= limit:
                break

        output = "\n".join(matches)
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"
        return ToolResult(success=True, output=output or "No matches.")
