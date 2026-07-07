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
