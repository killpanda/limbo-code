import tempfile
from pathlib import Path

import pytest

from limbo.tools.ls import LsTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "src").mkdir()
        (Path(tmp) / "README.md").write_text("# Hi")
        (Path(tmp) / ".env").write_text("x")
        yield Path(tmp)


def test_ls_lists_files_and_dirs(workdir):
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": "."})
    assert result.success is True
    assert "src/" in result.output
    assert "README.md" in result.output
    assert ".env" in result.output


def test_ls_limit(workdir):
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": ".", "limit": 2})
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 3  # 2 entries + truncation notice


def test_ls_truncation_notice(workdir):
    """Hitting the entry limit must never be silent (pi parity)."""
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": ".", "limit": 2})
    assert "Showing 2 of 3 entries" in result.output

    result = tool.execute({"path": ".", "limit": 10})
    assert "Showing" not in result.output


def test_ls_empty_directory(workdir):
    (workdir / "empty").mkdir()
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": "empty"})
    assert result.success is True
    assert result.output == "(empty directory)"


def test_ls_allows_paths_outside_workdir(workdir, tmp_path):
    """No workdir fence: directories outside the workdir are listable."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("x")
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": str(outside)})
    assert result.success is True
    assert "x.txt" in result.output


def test_ls_broken_symlink_returns_invalid_path(workdir):
    link = workdir / "broken_link"
    try:
        link.symlink_to("does_not_exist")
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": "broken_link"})
    assert result.success is False
    assert "invalid path" in result.error.lower()
