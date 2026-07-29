"""Gitignore matching for file tools.

Shared by ``find`` and ``grep``: a minimal matcher that reads the workdir
root ``.gitignore``. Does not support nested .gitignore files or advanced
glob/negation rules.
"""

from __future__ import annotations

import re
from pathlib import Path


class GitignoreMatcher:
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
