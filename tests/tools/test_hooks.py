"""Tests for the tool hook chain and the read-only plan mode hook."""

import tempfile
from pathlib import Path

import pytest

from limbo.models import ToolResult
from limbo.tools.hooks import (
    Block,
    PlanModeHook,
    Rewrite,
    ToolHook,
    is_read_only_command,
)
from limbo.tools.registry import ToolRegistry
from limbo.tools.run_code import RunCodeTool


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


def test_no_hooks_by_default(registry):
    assert registry.hooks == []
    assert registry.plan_mode is False


@pytest.mark.asyncio
async def test_hooks_see_every_call_in_order(workdir, registry):
    (workdir / "x.txt").write_text("hi")
    first, second = Recorder(), Recorder()
    registry.hooks += [first, second]
    result = await registry.execute("read", {"path": "x.txt"})
    assert result.success
    assert first.before == second.before == ["read"]
    assert first.after == second.after == ["read"]


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
async def test_rewrite_replaces_the_arguments(workdir, registry):
    (workdir / "a.txt").write_text("A")
    (workdir / "b.txt").write_text("B")
    registry.hooks.append(Recorder(Rewrite({"path": "b.txt"})))
    result = await registry.execute("read", {"path": "a.txt"})
    assert result.output == "B"


@pytest.mark.asyncio
async def test_a_raising_hook_blocks_the_call(workdir, registry):
    class Broken(ToolHook):
        name = "broken"

        def before_tool(self, name, arguments):
            raise RuntimeError("boom")

    registry.hooks.append(Broken())
    result = await registry.execute("write", {"path": "x.txt", "content": "a"})
    assert result.success is False
    assert "broken" in (result.error or "") and "boom" in (result.error or "")
    assert not (workdir / "x.txt").exists()


@pytest.mark.asyncio
async def test_after_hook_can_replace_the_result(workdir, registry):
    class Redact(ToolHook):
        def after_tool(self, name, arguments, result):
            return ToolResult(success=True, output="[redacted]")

    (workdir / "x.txt").write_text("secret")
    registry.hooks.append(Redact())
    result = await registry.execute("read", {"path": "x.txt"})
    assert result.output == "[redacted]"


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "cat a.txt | grep foo | wc -l",
        "git status && git log --oneline -5",
        "git -C sub diff HEAD~1",
        "git branch -a",
        "rg TODO src 2>/dev/null",
        "find . -name '*.py'",
        "sed -n 1,20p x.py",
        "grep -r foo . 2>&1 | head",
    ],
)
def test_read_only_commands_are_allowed(command):
    assert is_read_only_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "echo hi > out.txt",
        "cat a >> b",
        "ls && touch x",
        "git commit -m x",
        "git branch -D old",
        "git branch new-branch",
        "sed -i s/a/b/ x.py",
        "find . -name '*.pyc' -delete",
        "find . -exec rm {} ;",
        "echo $(rm x)",
        "python script.py",
        "ls &> out.txt",
        "unterminated 'quote",
    ],
)
def test_state_changing_commands_are_blocked(command):
    assert not is_read_only_command(command)


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


def test_plan_mode_toggle_is_idempotent(registry):
    registry.plan_mode = True
    registry.plan_mode = True
    assert sum(isinstance(h, PlanModeHook) for h in registry.hooks) == 1


def test_plan_mode_applies_to_code_mode_sub_calls(workdir, registry):
    registry.plan_mode = True
    tool = RunCodeTool(workdir=registry.workdir)
    tool.registry = registry
    result = tool.run(
        {
            "code": (
                "try:\n"
                '    await tools.write(path="y.txt", content="a")\n'
                "except ToolCallError as e:\n"
                '    return f"blocked {e.toolName}"'
            ),
            "description": "Try to write in plan mode",
        }
    )
    assert result.success is True
    assert "blocked write" in (result.output or "")
    assert not (workdir / "y.txt").exists()
