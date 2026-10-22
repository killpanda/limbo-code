"""Tests for the run_code tool (Code Mode transport, deepseek-harness parity)."""

import tempfile
from pathlib import Path

import pytest

from limbo.tools.registry import ToolRegistry
from limbo.tools.run_code import RunCodeTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def registry(workdir):
    return ToolRegistry(workdir=workdir)


@pytest.fixture
def tool(registry):
    t = RunCodeTool(workdir=registry.workdir)
    t.registry = registry
    return t


def test_gather_parallel_calls(workdir, tool):
    (workdir / "a.txt").write_text("aaa")
    (workdir / "b.txt").write_text("bbb")
    result = tool.run(
        {
            "code": (
                "import asyncio\n"
                "a, b = await asyncio.gather(\n"
                '    tools.read(path="a.txt"),\n'
                '    tools.read(path="b.txt"),\n'
                ")\n"
                'return a + "|" + b'
            ),
            "description": "Read two files concurrently",
        }
    )
    assert result.success is True
    assert "aaa|bbb" in (result.output or "")


