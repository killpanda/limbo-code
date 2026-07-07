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
    result = tool.execute(
        {"command": "python -c \"import sys; sys.stderr.write('error\\n')\""}
    )
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


def test_bash_rejects_dangerous_command(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "rm -rf /"})
    assert result.success is False
    assert "blocked" in result.error.lower()


def test_bash_workdir_context(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "pwd"})
    assert result.success is True
    assert str(workdir) in result.output


def test_bash_safety_filter_is_heuristic(workdir):
    """Document that the simple pattern matcher can be bypassed."""
    tool = BashTool(workdir=workdir)
    # Subshell / command substitution bypasses the top-level token check.
    result = tool.execute({"command": "bash -c 'echo bypassed'"})
    assert result.success is True
    assert "bypassed" in result.output
    assert "blocked" not in (result.error or "").lower()


def test_is_dangerous_docstring_warns_about_bypasses():
    assert "heuristic" in (is_dangerous.__doc__ or "").lower()
    assert "bypass" in (is_dangerous.__doc__ or "").lower()


def test_is_dangerous_allows_redirection_in_quoted_string():
    tool = BashTool(workdir=Path("/tmp"))
    result = tool.execute({"command": 'echo "a > b"'})
    assert result.success is True
    assert "blocked" not in (result.error or "").lower()


def test_is_dangerous_blocks_actual_redirection():
    tool = BashTool(workdir=Path("/tmp"))
    result = tool.execute({"command": "echo a > /tmp/limbo-test-redir.txt"})
    assert result.success is False
    assert "blocked" in result.error.lower()


def test_is_dangerous_blocks_git_reset_hard():
    tool = BashTool(workdir=Path("/tmp"))
    result = tool.execute({"command": "git reset --hard HEAD"})
    assert result.success is False
    assert "blocked" in result.error.lower()


def test_is_dangerous_allows_git_status(workdir):
    import subprocess

    subprocess.run(["git", "init", str(workdir)], check=True, capture_output=True)
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "git status"})
    assert "blocked" not in (result.error or "").lower()
    assert result.success is True


def test_is_dangerous_blocks_absolute_rm_path():
    tool = BashTool(workdir=Path("/tmp"))
    result = tool.execute({"command": "/bin/rm -rf /tmp/limbo-fake-target"})
    assert result.success is False
    assert "blocked" in result.error.lower()


def test_is_dangerous_still_bypassed_by_subshell():
    tool = BashTool(workdir=Path("/tmp"))
    result = tool.execute({"command": "bash -c 'echo harmless'"})
    assert result.success is True
    assert "blocked" not in (result.error or "").lower()


def test_bash_custom_dangerous_patterns(workdir):
    tool = BashTool(workdir=workdir, dangerous_patterns=["reboot"])
    result = tool.execute({"command": "reboot"})
    assert result.success is False
    assert "blocked" in result.error.lower()


def test_bash_allows_command_not_in_custom_patterns(workdir):
    tool = BashTool(workdir=workdir, dangerous_patterns=["reboot"])
    result = tool.execute({"command": "echo hello"})
    assert result.success is True


def test_bash_dry_run_does_not_execute(workdir):
    tool = BashTool(workdir=workdir)
    marker = workdir / "dry-run-marker.txt"
    result = tool.execute(
        {"command": f"touch {marker}"}, dry_run=True
    )
    assert result.success is True
    assert result.requires_confirmation is True
    assert "dry-run-marker" in result.output
    assert "approval" in result.output.lower()
    assert not marker.exists()


def test_bash_dry_run_blocks_dangerous_command(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "rm -rf /"}, dry_run=True)
    assert result.success is False
    assert "blocked" in result.error.lower()
