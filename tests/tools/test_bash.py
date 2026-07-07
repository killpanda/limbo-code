import tempfile
from pathlib import Path

import pytest

from limbo.tools.bash import BashTool, is_dangerous


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_bash_echo(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo hello"})
    assert result.success is True
    assert "hello" in result.output


def test_bash_captures_stderr(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo error >&2"})
    assert result.success is True
    assert "error" in result.output


def test_bash_failure_returns_non_zero(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "exit 42"})
    assert result.success is False
    assert "exit code 42" in result.error.lower()


def test_bash_timeout(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "sleep 5", "timeout": 0.1})
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_is_dangerous_detects_rm():
    assert is_dangerous("rm -rf /", ["rm"]) is True
    assert is_dangerous("echo hello", ["rm"]) is False


def test_bash_workdir_context(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "pwd"})
    assert result.success is True
    assert str(workdir) in result.output
