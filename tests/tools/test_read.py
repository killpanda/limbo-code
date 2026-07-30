import tempfile
from pathlib import Path

import pytest

from limbo.tools.read import MAX_FILE_SIZE, ReadTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_read_success(workdir):
    (workdir / "hello.txt").write_text("hello world")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "hello.txt"})
    assert result.success is True
    assert result.output == "hello world"


def test_read_with_offset_and_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\nline3\nline4\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": 2, "limit": 2})
    assert result.success is True
    assert result.output == "line2\nline3\n"


def test_read_outside_workdir(workdir):
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "../etc/passwd"})
    assert result.success is False
    assert "outside working directory" in result.error.lower()


def test_read_sensitive_file(workdir):
    (workdir / ".env").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": ".env"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_read_custom_sensitive_files(workdir):
    (workdir / "secret.txt").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir, sensitive_files=["secret.txt"])
    result = tool.execute({"path": "secret.txt"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_read_non_sensitive_file_not_in_custom_list(workdir):
    (workdir / ".env").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir, sensitive_files=["secret.txt"])
    result = tool.execute({"path": ".env"})
    assert result.success is True


def test_read_blocks_ssh_path(workdir):
    ssh_dir = workdir / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Host example")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": ".ssh/config"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_read_rejects_negative_offset(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": -1})
    assert result.success is False
    assert "positive" in result.error.lower()


def test_read_rejects_negative_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "limit": -1})
    assert result.success is False
    assert "positive" in result.error.lower()


def test_read_does_not_follow_symlink_to_outside_file(workdir):
    """A symlink inside the workdir must not expose files outside it."""
    import os

    outside = workdir.parent / f"read_outside_{workdir.name}"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")

    link = workdir / "outside_link"
    try:
        os.symlink(outside / "secret.txt", link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "outside_link"})
    assert result.success is False
    assert "outside working directory" in result.error.lower()


def test_read_broken_symlink_returns_invalid_path(workdir):
    """A broken symlink causes resolve() to raise OSError; the tool must handle it."""
    import os

    link = workdir / "broken_link"
    try:
        os.symlink("does_not_exist", link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "broken_link"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_read_symlink_loop_returns_invalid_path(workdir):
    """A symlink loop causes resolve() to raise RuntimeError; the tool must handle it."""
    import os

    link = workdir / "loop"
    try:
        os.symlink("loop", link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "loop"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_read_truncates_output_by_byte_budget(workdir):
    """Multi-byte characters count against the byte budget, not character count."""
    (workdir / "big.txt").write_text("é" * (60 * 1024))  # 120 KB of UTF-8 bytes
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "big.txt"})
    assert result.success is True
    assert len(result.output.encode("utf-8")) <= 50 * 1024 + 200
    assert "truncated" in result.output.lower()
    assert "Use offset/limit to read more" in result.output


def test_read_truncates_output_by_line_budget(workdir):
    """Head truncation keeps the first 2000 lines and reports the total."""
    (workdir / "lines.txt").write_text("\n".join(f"line-{i}" for i in range(1, 3001)))
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt"})
    assert result.success is True
    assert "line-2000" in result.output
    assert "line-2001" not in result.output
    assert "showing lines 1-2000 of 3000" in result.output
    assert "Use offset/limit to read more" in result.output


def test_read_truncation_hint_uses_absolute_line_numbers(workdir):
    """With offset, the hint must refer to file line numbers, not slice-relative ones."""
    (workdir / "lines.txt").write_text("\n".join(f"line-{i}" for i in range(1, 3001)))
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": 500})
    assert result.success is True
    assert "line-500" in result.output
    assert "line-2499" in result.output
    assert "line-2500" not in result.output
    assert "showing lines 500-2499 of 3000" in result.output


def test_read_allows_offloaded_bash_output_in_tmpdir(workdir):
    """Bash offloading promises the model can retrieve full output with read."""
    offload = Path(tempfile.gettempdir()) / "limbo-output-testdeadbeef.log"
    offload.write_text("full bash output")
    try:
        tool = ReadTool(workdir=workdir)
        result = tool.execute({"path": str(offload)})
        assert result.success is True
        assert result.output == "full bash output"
    finally:
        offload.unlink(missing_ok=True)


def test_read_offload_whitelist_rejects_other_tmp_paths(workdir):
    """Only limbo-output-*.log directly inside tmpdir is whitelisted."""
    tmpdir = Path(tempfile.gettempdir())
    victims = [
        tmpdir / "other.log",  # not a limbo offload name
        tmpdir / "limbo-output-x.txt",  # wrong extension
    ]
    sub = tmpdir / "limbo-sub"
    sub.mkdir(exist_ok=True)
    victims.append(sub / "limbo-output-x.log")  # not directly inside tmpdir
    try:
        for victim in victims:
            victim.write_text("secret")
        tool = ReadTool(workdir=workdir)
        for victim in victims:
            result = tool.execute({"path": str(victim)})
            assert result.success is False, victim
            assert "outside working directory" in (result.error or "").lower()
    finally:
        for victim in victims:
            victim.unlink(missing_ok=True)
        sub.rmdir()


def test_read_rejects_files_above_max_size(workdir):
    """Files larger than the generous cap must be refused before reading."""
    big = workdir / "huge.txt"
    big.write_text("x" * (MAX_FILE_SIZE + 1))
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "huge.txt"})
    assert result.success is False
    assert "too large" in result.error.lower()


def test_granted_root_allows_read_outside_workdir(workdir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("external")
    shared: set[Path] = set()
    tool = ReadTool(workdir=workdir, allowed_roots=shared)
    result = tool.execute({"path": str(outside / "data.txt")})
    assert result.success is False
    shared.add(outside.resolve())
    result = tool.execute({"path": str(outside / "data.txt")})
    assert result.success is True
    assert result.output == "external"


def test_sensitive_blacklist_wins_over_granted_root(workdir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".env").write_text("SECRET=1")
    shared: set[Path] = {outside.resolve()}
    tool = ReadTool(workdir=workdir, allowed_roots=shared)
    result = tool.execute({"path": str(outside / ".env")})
    assert result.success is False
    assert "sensitive" in result.error
