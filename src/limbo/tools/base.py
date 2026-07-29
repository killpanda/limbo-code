"""Base class and shared helpers for tools.

The base module owns the rituals every tool used to repeat: workdir-safe
path resolution (raising ``ToolError`` instead of returning union types)
and the output truncation policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from limbo.models import ToolResult

MAX_OUTPUT_BYTES = 512 * 1024


class ToolError(Exception):
    """Raised by tool helpers; ``BaseTool.execute`` turns it into an error result."""


class BaseTool(ABC):
    """Abstract base class for all Limbo tools.

    Subclasses implement ``run``; ``execute`` is the public entry point and
    converts ``ToolError`` into an error ``ToolResult`` uniformly.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return self.run(arguments)
        except ToolError as e:
            return ToolResult(success=False, error=str(e))

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        ...

    # -- path resolution -------------------------------------------------------

    def resolve(self, raw_path: str, *, strict: bool = True) -> Path:
        """Resolve ``raw_path`` under the workdir, enforcing the boundary.

        Raises ``ToolError`` for paths outside the workdir, unresolvable
        paths, and (when ``strict``) broken symlinks.
        """
        raw = self.workdir / raw_path
        try:
            target = raw.resolve(strict=False)
        except (OSError, RuntimeError) as e:
            raise ToolError(f"Invalid path: {e}") from e

        if not is_within_workdir(target, self.workdir):
            raise ToolError("Path is outside working directory.")

        if strict and raw.is_symlink() and not target.exists():
            raise ToolError("Invalid path: broken symlink")

        return target

    def resolve_existing(
        self, raw_path: str, *, noun: str = "Path", kind: str = "any"
    ) -> Path:
        """Resolve a path that must exist. ``kind``: any | file | dir."""
        target = self.resolve(raw_path)
        if not target.exists():
            raise ToolError(f"{noun} not found: {raw_path}")
        if kind == "file" and not target.is_file():
            raise ToolError(f"Not a file: {raw_path}")
        if kind == "dir" and not target.is_dir():
            raise ToolError(f"Not a directory: {raw_path}")
        return target

    def resolve_creatable(self, raw_path: str) -> Path:
        """Resolve a path that may not exist yet (e.g. for writing)."""
        return self.resolve(raw_path, strict=False)

def is_within_workdir(path: Path, workdir: Path) -> bool:
    """Return True if resolved path is inside or equal to workdir."""
    try:
        path.resolve().relative_to(workdir.resolve())
        return True
    except ValueError:
        return False


def truncate_output(
    output: str, *, limit: int = MAX_OUTPUT_BYTES, suffix: str = "\n[Output truncated.]"
) -> str:
    """Truncate ``output`` to ``limit`` bytes (UTF-8 safe) and append ``suffix``."""
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return output
    return encoded[:limit].decode("utf-8", errors="replace") + suffix
