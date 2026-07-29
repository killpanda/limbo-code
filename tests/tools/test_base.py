"""Tests for the BaseTool seam: resolution helpers, ToolError."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError, truncate_output


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


def test_execute_converts_tool_error_to_error_result(tool):
    result = tool.execute({"action": "raise"})
    assert result.success is False
    assert result.error == "boom"


def test_resolve_existing_returns_path(tool):
    result = tool.execute({"action": "existing"})
    assert result.success is True
    assert result.output.endswith("f.txt")


def test_resolve_existing_missing_file(tool):
    tool_with_args = tool
    try:
        tool_with_args.resolve_existing("missing.txt")
    except ToolError as e:
        assert "Path not found: missing.txt" == str(e)
    else:
        raise AssertionError("expected ToolError")


def test_resolve_existing_kind_checks(tool, tmp_path):
    (tmp_path / "subdir").mkdir()
    with pytest.raises(ToolError, match="Not a file"):
        tool.resolve_existing("subdir", kind="file")
    with pytest.raises(ToolError, match="Not a directory"):
        tool.resolve_existing("f.txt", kind="dir")


def test_resolve_rejects_outside_workdir(tool):
    with pytest.raises(ToolError, match="outside working directory"):
        tool.resolve("../escape.txt")


def test_resolve_creatable_allows_missing(tool):
    result = tool.execute({"action": "creatable"})
    assert result.success is True
    assert result.output.endswith("new.txt")


def test_truncate_output_by_bytes():
    text = "x" * 100
    out = truncate_output(text, limit=10)
    assert out == "x" * 10 + "\n[Output truncated.]"
    assert truncate_output(text, limit=1000) == text
