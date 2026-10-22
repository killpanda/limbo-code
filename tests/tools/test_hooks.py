"""Tests for the tool hook chain and the read-only plan mode hook."""

import tempfile
from pathlib import Path

import pytest

from limbo.tools.hooks import (
    Block,
    ToolHook,
)
from limbo.tools.registry import ToolRegistry


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def registry(workdir):
    return ToolRegistry(workdir=workdir)


class Recorder(ToolHook):
    name = "recorder"

    def __init__(self, decision=None):
        self.decision = decision
        self.before: list[str] = []
        self.after: list[str] = []

    def before_tool(self, name, arguments):
        self.before.append(name)
        return self.decision

    def after_tool(self, name, arguments, result):
        self.after.append(name)
        return None


@pytest.mark.asyncio
async def test_block_skips_the_tool_and_later_hooks(workdir, registry):
    later = Recorder()
    registry.hooks += [Recorder(Block("nope")), later]
    result = await registry.execute("write", {"path": "x.txt", "content": "a"})
    assert result.success is False
    assert result.error == "nope"
    assert not (workdir / "x.txt").exists()
    assert later.before == []


@pytest.mark.asyncio
async def test_plan_mode_blocks_writes_and_allows_reads(workdir, registry):
    (workdir / "x.txt").write_text("hi")
    registry.plan_mode = True
    assert registry.plan_mode
    assert (await registry.execute("read", {"path": "x.txt"})).success
    assert (await registry.execute("ls", {})).success
    blocked = await registry.execute("write", {"path": "y.txt", "content": "a"})
    assert blocked.success is False
    assert "Plan mode" in (blocked.error or "")
    assert not (workdir / "y.txt").exists()
    edit = await registry.execute(
        "edit", {"path": "x.txt", "edits": [{"old_text": "hi", "new_text": "yo"}]}
    )
    assert edit.success is False
    assert (workdir / "x.txt").read_text() == "hi"
    bash = await registry.execute("bash", {"command": "touch z.txt"})
    assert bash.success is False
    assert not (workdir / "z.txt").exists()
    assert (await registry.execute("bash", {"command": "ls"})).success

    registry.plan_mode = False
    assert registry.hooks == []
    assert (await registry.execute("write", {"path": "y.txt", "content": "a"})).success


