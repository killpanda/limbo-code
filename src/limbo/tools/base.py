"""Base class and helpers for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from limbo.models import ToolResult


class BaseTool(ABC):
    """Abstract base class for all Limbo tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()

    @abstractmethod
    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        ...


def is_within_workdir(path: Path, workdir: Path) -> bool:
    """Return True if resolved path is inside or equal to workdir."""
    try:
        path.resolve().relative_to(workdir.resolve())
        return True
    except ValueError:
        return False


def resolve_path(
    raw_path: str, workdir: Path, *, strict: bool = True
) -> Path | ToolResult:
    """Resolve ``raw_path`` under ``workdir`` and ensure it stays within it.

    Returns the resolved ``Path`` on success, or a ``ToolResult`` error on
    failure. Catches ``OSError`` and ``RuntimeError`` (e.g. symlink loops) from
    ``Path.resolve()``. When ``strict`` is ``True``, a broken symlink is also
    reported as an invalid path so that every file tool returns a clean error
    instead of leaking a raw exception.
    """
    raw = workdir / raw_path
    try:
        target = raw.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        return ToolResult(success=False, error=f"Invalid path: {e}")

    if not is_within_workdir(target, workdir):
        return ToolResult(success=False, error="Path is outside working directory.")

    if strict and raw.is_symlink() and not target.exists():
        return ToolResult(success=False, error="Invalid path: broken symlink")

    return target
