"""Tests for the BaseTool seam: resolution helpers, ToolError."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from limbo.models import ToolResult
from limbo.tools.base import (
    BaseTool,
    ToolError,
    truncate_head,
    truncate_output,
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


def test_resolve_allows_outside_workdir(tool):
    """No workdir fence: relative escapes and absolute paths resolve as-is."""
    resolved = tool.resolve("../escape.txt")
    assert resolved == (tool.workdir / "../escape.txt").resolve()


def test_resolve_expands_tilde(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workdir = home / "proj"
    workdir.mkdir(parents=True)
    (workdir / "f.txt").write_text("x")
    monkeypatch.setenv("HOME", str(home))
    tool = DummyTool(workdir)
    # ~/proj/f.txt lands inside the workdir once ~ is expanded.
    assert tool.resolve("~/proj/f.txt") == workdir / "f.txt"


def test_resolve_creatable_allows_missing(tool):
    result = tool.execute({"action": "creatable"})
    assert result.success is True
    assert result.output.endswith("new.txt")


def test_truncate_output_by_bytes():
    text = "x" * 100
    out = truncate_output(text, limit=10)
    assert out == "x" * 10 + "\n[Output truncated.]"
    assert truncate_output(text, limit=1000) == text


def _numberd_lines(n):
    return "\n".join(str(i) for i in range(1, n + 1))


def test_truncate_head_keeps_first_lines():
    result = truncate_head(_numberd_lines(3000))
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.output_lines == 2000
    assert result.total_lines == 3000
    assert result.start_line == 1
    lines = result.content.split("\n")
    assert lines[0] == "1"
    assert lines[-1] == "2000"


def test_truncate_tail_keeps_last_lines():
    result = truncate_tail(_numberd_lines(3000))
    assert result.truncated is True
    assert result.truncated_by == "lines"
    assert result.output_lines == 2000
    assert result.start_line == 1001
    lines = result.content.split("\n")
    assert lines[0] == "1001"
    assert lines[-1] == "3000"


def test_truncate_head_respects_byte_budget():
    # 1000 lines x 100 bytes each: bytes bite before lines.
    result = truncate_head("\n".join(["x" * 100] * 1000), max_bytes=5000)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert len(result.content.encode("utf-8")) <= 5000
    assert "\n" in result.content  # never returns partial lines


def test_truncate_tail_respects_byte_budget_and_keeps_end():
    lines = ["x" * 100] * 999 + ["final"]
    result = truncate_tail("\n".join(lines), max_bytes=5000)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.content.split("\n")[-1] == "final"
    assert len(result.content.encode("utf-8")) <= 5000


def test_truncate_tail_single_oversized_line_keeps_byte_suffix():
    result = truncate_tail("x" * 100 + "tail", max_bytes=10)
    assert result.truncated is True
    assert result.truncated_by == "bytes"
    assert result.edge_line_partial is True
    assert result.content.endswith("tail")
    assert len(result.content.encode("utf-8")) <= 10


def test_truncate_head_single_oversized_line_keeps_byte_prefix():
    result = truncate_head("head" + "x" * 100, max_bytes=10)
    assert result.truncated is True
    assert result.edge_line_partial is True
    assert result.content.startswith("head")
    assert len(result.content.encode("utf-8")) <= 10


def test_truncate_helpers_pass_through_small_content():
    text = "hello\nworld\n"
    for fn in (truncate_head, truncate_tail):
        result = fn(text)
        assert result.truncated is False
        assert result.truncated_by is None
        assert result.content == text
        assert result.total_lines == 2


def test_truncate_multibyte_characters_count_bytes():
    # Each 'é' is two UTF-8 bytes.
    result = truncate_tail("é" * 100, max_bytes=50)
    assert result.truncated is True
    assert len(result.content.encode("utf-8")) <= 50
    # No replacement characters: the cut respects UTF-8 boundaries.
    assert "\ufffd" not in result.content
