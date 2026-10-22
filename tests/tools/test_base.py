"""Tests for the BaseTool seam: resolution helpers, ToolError."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from limbo.models import ToolResult
from limbo.tools.base import (
    BaseTool,
    ToolError,
    truncate_tail,
)


class DummyTool(BaseTool):
    name = "dummy"
    description = "dummy"
    parameters: dict[str, Any] = {}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        action = arguments.get("action")
        if action == "raise":
            raise ToolError("boom")
        if action == "existing":
            return ToolResult(
                success=True, output=str(self.resolve_existing("f.txt"))
            )
        if action == "creatable":
            return ToolResult(
                success=True, output=str(self.resolve_creatable("new.txt"))
            )
        return ToolResult(success=False, error="unknown")


@pytest.fixture
def tool(tmp_path: Path) -> DummyTool:
    (tmp_path / "f.txt").write_text("x")
    return DummyTool(tmp_path)


def test_truncate_tail_respects_byte_budget_and_keeps_end():
    lines = ["x" * 100] * 999 + ["final"]
    result = truncate_tail("\n".join(lines), max_bytes=5000)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content.split("\n")[-1] == "final"
    assert len(result.content.encode("utf-8")) <= 5000


