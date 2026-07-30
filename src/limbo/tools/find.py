"""Find files by glob pattern tool."""

from __future__ import annotations

from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import (
    BaseTool,
    truncate_output,
)
from limbo.tools.ignore import GitignoreMatcher

MAX_RESULTS = 1000


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

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_RESULTS)

        target = self.resolve_existing(path)

        try:
            # ``glob`` may traverse directory symlinks before we can filter them;
            # we resolve and enforce the workdir boundary below.
            matches = sorted(target.glob(pattern))
        except ValueError as e:
            return ToolResult(success=False, error=f"Invalid glob pattern: {e}")

        matcher = GitignoreMatcher(self.workdir)

        results = []
        for p in matches:
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if not resolved.is_file():
                continue
            if not self.is_within_scope(resolved):
                continue
            try:
                rel = str(p.relative_to(self.workdir))
            except ValueError:
                # Inside a granted root outside the workdir: show the
                # absolute path (gitignore matching only applies under the
                # workdir).
                results.append(str(resolved))
            else:
                if matcher.is_ignored(rel):
                    continue
                results.append(rel)
            if len(results) >= limit:
                break

        output = "\n".join(results)
        output = truncate_output(output)

        return ToolResult(success=True, output=output or "No matches.")
