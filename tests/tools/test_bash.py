import re
import tempfile
import time
from pathlib import Path

import pytest

from limbo.tools.bash import BashTool


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
    assert "exit code 42" in result.output.lower()


def test_bash_failure_includes_exit_code_with_output(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo partial && exit 7"})
    assert result.success is False
    assert "exit code 7" in result.output.lower()
    assert "partial" in result.output


def test_bash_timeout(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "sleep 5", "timeout": 0.1})
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_bash_timeout_returns_partial_output(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo before-sleep && sleep 5", "timeout": 0.2})
    assert result.success is False
    assert "timed out" in result.error.lower()
    assert "before-sleep" in result.output


def test_bash_timeout_kills_process_tree(workdir):
    """A timed-out command's whole tree is killed (pi's killProcessTree).

    With a detached grandchild that keeps the pipe open, a plain
    ``proc.kill()`` would leave the reader threads blocked on the pipe
    and the call hanging; killing the process group tears it all down.
    """
    tool = BashTool(workdir=workdir)
    start = time.monotonic()
    result = tool.execute({"command": "sleep 60 & sleep 60", "timeout": 0.2})
    elapsed = time.monotonic() - start
    assert result.success is False
    assert "timed out" in result.error.lower()
    assert elapsed < 10


def test_bash_no_default_timeout(workdir):
    """Without an explicit timeout a command runs to completion."""
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "sleep 1 && echo done"})
    assert result.success is True
    assert "done" in result.output


def test_bash_accepts_large_timeout(workdir):
    """No hardcoded cap: huge timeouts are honored, not clamped."""
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo ok", "timeout": 1_000_000})
    assert result.success is True
    assert result.output.strip() == "ok"


def test_bash_rejects_invalid_timeout(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo ok", "timeout": "not-a-number"})
    assert result.success is False
    assert "invalid timeout" in result.error.lower()


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_bash_rejects_non_positive_or_non_finite_timeout(workdir, bad):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo ok", "timeout": bad})
    assert result.success is False
    assert "invalid timeout" in result.error.lower()


def test_bash_no_command_filter(workdir):
    """No dangerous-command filter: even pattern-listed commands run."""
    target = workdir / "doomed"
    target.mkdir()
    (target / "f.txt").write_text("x")
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": f"rm -rf {target}"})
    assert result.success is True
    assert not target.exists()


def test_bash_workdir_context(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "pwd"})
    assert result.success is True
    assert str(workdir) in result.output


def test_bash_allows_redirection_in_quoted_string(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": 'echo "a > b"'})
    assert result.success is True
    assert "a > b" in result.output


def test_bash_allows_redirection(workdir):
    target = workdir / "limbo-test-redir.txt"
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": f"echo a > {target}"})
    assert result.success is True
    assert target.read_text().strip() == "a"


def test_bash_allows_git_status(workdir):
    import subprocess

    subprocess.run(["git", "init", str(workdir)], check=True, capture_output=True)
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "git status"})
    assert result.success is True


def test_bash_unbalanced_quotes_returns_clean_error(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": 'echo "unbalanced'})
    # bash itself reports the quoting error; no raw ValueError escapes.
    assert result.success is False
    assert "No closing quotation" not in (result.error or "")


def test_bash_description_does_not_teach_bypasses():
    """The tool description must not enumerate filter-bypass techniques (P0-3a)."""
    description = BashTool.description.lower()
    for phrase in ("subshell", "command substitution", "variable indirection", "bypass"):
        assert phrase not in description
    assert "not sandboxed" in description


def test_bash_description_steers_scripts_to_write_tool():
    """S4: the description pre-empts long heredoc one-liners."""
    description = BashTool.description.lower()
    assert "heredoc" in description
    assert "write" in description


def test_bash_handles_non_utf8_output(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute(
        {"command": "python -c \"import sys; sys.stdout.buffer.write(b'\\\\xff\\\\xfe')\""}
    )
    assert result.success is True
    assert "\ufffd" in result.output


def _extract_full_output_path(output: str) -> Path:
    match = re.search(r"Full output: (\S+)\]", output)
    assert match is not None, f"no Full output path in: {output[-200:]}"
    return Path(match.group(1))


def test_bash_truncates_excessive_output(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute(
        {"command": "python -c \"print('x' * (600 * 1024), end='')\""}
    )
    assert result.success is True
    assert len(result.output.encode("utf-8")) <= 50 * 1024 + 200
    assert "Showing lines" in result.output
    full_path = _extract_full_output_path(result.output)
    try:
        assert full_path.name.startswith("limbo-output-")
        assert len(full_path.read_text(encoding="utf-8").encode("utf-8")) >= 600 * 1024
    finally:
        full_path.unlink(missing_ok=True)


def test_bash_truncates_output_by_byte_budget(workdir):
    """Multi-byte characters count against the byte budget, not character count."""
    tool = BashTool(workdir=workdir)
    # Each 'é' is two UTF-8 bytes; 30 KiB of characters is 60 KiB of bytes.
    result = tool.execute(
        {"command": "python -c \"import sys; sys.stdout.write('é' * (30 * 1024))\""}
    )
    assert result.success is True
    assert len(result.output.encode("utf-8")) <= 50 * 1024 + 200
    assert "Showing lines" in result.output
    full_path = _extract_full_output_path(result.output)
    try:
        assert len(full_path.read_text(encoding="utf-8").encode("utf-8")) == 60 * 1024
    finally:
        full_path.unlink(missing_ok=True)


def test_bash_truncates_by_line_budget_and_keeps_tail(workdir):
    """Excess lines: keep the tail (errors/results live at the end)."""
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "seq 1 3000"})
    assert result.success is True
    kept = result.output.split("\n")
    # Tail lines are kept; head lines are dropped.
    assert "3000" in kept
    assert "2999" in kept
    assert "1" not in kept[:50]
    assert "[Showing lines 1001-3000 of 3000. Full output:" in result.output
    full_path = _extract_full_output_path(result.output)
    try:
        full_lines = full_path.read_text(encoding="utf-8").splitlines()
        assert full_lines[0] == "1"
        assert full_lines[-1] == "3000"
        assert len(full_lines) == 3000
    finally:
        full_path.unlink(missing_ok=True)


def test_bash_small_output_is_not_offloaded(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo hello"})
    assert result.success is True
    assert "Full output:" not in result.output
    assert "Showing lines" not in result.output


def test_bash_failed_command_still_offloads_full_output(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "seq 1 3000 && exit 3"})
    assert result.success is False
    assert "exit code 3" in result.error.lower()
    assert "[Showing lines 1001-3000 of 3000. Full output:" in result.output
    full_path = _extract_full_output_path(result.output)
    try:
        assert len(full_path.read_text(encoding="utf-8").splitlines()) == 3000
    finally:
        full_path.unlink(missing_ok=True)
