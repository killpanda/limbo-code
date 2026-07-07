"""Find files by glob pattern tool."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir, resolve_path

MAX_RESULTS = 1000
MAX_BYTES = 512 * 1024


class _GitignoreMatcher:
    """Minimal gitignore matcher that reads the workdir root `.gitignore`."""

    def __init__(self, base_dir: Path) -> None:
        self.rules: list[tuple[bool, re.Pattern[str], bool]] = []
        gitignore = base_dir / ".gitignore"
        if gitignore.is_file():
            try:
                text = gitignore.read_text(encoding="utf-8")
            except OSError:
                text = ""
            for line in text.splitlines():
                rule = self._parse_line(line)
                if rule is not None:
                    self.rules.append(rule)

    def _parse_line(self, line: str) -> tuple[bool, re.Pattern[str], bool] | None:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            return None

        negation = False
        if line.startswith("!"):
            negation = True
            line = line[1:]
            if not line:
                return None
        elif line.startswith("\\!"):
            line = line[1:]

        if line.startswith("#"):
            return None

        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]

        anchored = "/" in line
        if line.startswith("/"):
            line = line[1:]

        regex = self._glob_to_regex(line)
        if not anchored:
            regex = f"(?:.*/)?{regex}"
        pattern = re.compile(f"^{regex}$")
        return (negation, pattern, dir_only)

    def _glob_to_regex(self, pattern: str) -> str:
        i = 0
        n = len(pattern)
        result: list[str] = []
        while i < n:
            c = pattern[i]
            if c == "*" and i + 1 < n and pattern[i + 1] == "*":
                result.append(".*")
                i += 2
            elif c == "*":
                result.append("[^/]*")
                i += 1
            elif c == "?":
                result.append("[^/]")
                i += 1
            elif c == "[":
                j = i + 1
                if j < n and pattern[j] == "!":
                    j += 1
                if j < n and pattern[j] == "]":
                    j += 1
                while j < n and pattern[j] != "]":
                    j += 1
                if j >= n:
                    result.append(re.escape(c))
                    i += 1
                else:
                    chars = pattern[i + 1 : j]
                    if chars.startswith("!"):
                        chars = f"^{chars[1:]}"
                    result.append(f"[{chars}]")
                    i = j + 1
            else:
                result.append(re.escape(c))
                i += 1
        return "".join(result)

    def is_ignored(self, rel_path: str) -> bool:
        parts = rel_path.split("/")
        paths = [rel_path]
        paths.extend("/".join(parts[:k]) for k in range(1, len(parts)))

        ignored = False
        for negation, regex, dir_only in self.rules:
            for path in paths:
                if dir_only and path == rel_path:
                    continue
                if regex.match(path):
                    ignored = not negation
                    break
        return ignored


class FindTool(BaseTool):
    name = "find"
    description = (
        "Find files by glob pattern. Respects the root .gitignore only. "
        "The built-in matcher is minimal and does not support nested .gitignore "
        "files or advanced glob/negation rules."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["pattern"],
    }

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_RESULTS)

        target = resolve_path(path, self.workdir)
        if isinstance(target, ToolResult):
            return target
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            # ``glob`` may traverse directory symlinks before we can filter them;
            # we resolve and enforce the workdir boundary below.
            matches = sorted(target.glob(pattern))
        except ValueError as e:
            return ToolResult(success=False, error=f"Invalid glob pattern: {e}")

        matcher = _GitignoreMatcher(self.workdir)

        results = []
        for p in matches:
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if not resolved.is_file():
                continue
            if not is_within_workdir(resolved, self.workdir):
                continue
            rel = str(p.relative_to(self.workdir))
            if matcher.is_ignored(rel):
                continue
            results.append(rel)
            if len(results) >= limit:
                break

        output = "\n".join(results)
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"

        return ToolResult(success=True, output=output or "No matches.")
