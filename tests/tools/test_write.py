import tempfile
from pathlib import Path

import pytest

from limbo.tools.write import WriteTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_write_creates_file(workdir):
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "new.txt", "content": "hello"})
    assert result.success is True
    assert (workdir / "new.txt").read_text() == "hello"


def test_write_overwrites_file(workdir):
    (workdir / "x.txt").write_text("old")
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "x.txt", "content": "new"})
    assert result.success is True
    assert (workdir / "x.txt").read_text() == "new"


def test_write_outside_workdir(workdir, tmp_path):
    """No workdir fence: paths outside the workdir are writable."""
    outside = tmp_path / "outside.txt"
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": str(outside), "content": "x"})
    assert result.success is True
    assert outside.read_text() == "x"


def test_write_follows_symlink_to_outside(workdir, tmp_path):
    """A symlink pointing outside the workdir resolves to its target."""
    outside = tmp_path / "outside_write_target.txt"
    link = workdir / "escape_link"
    link.symlink_to(outside)
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "escape_link", "content": "escaped"})
    assert result.success is True
    assert outside.read_text() == "escaped"


def test_write_broken_symlink_to_outside_creates_target(workdir, tmp_path):
    """A broken symlink resolves (non-strict) to its outside target."""
    outside = tmp_path / "created_target.txt"
    link = workdir / "broken_link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "broken_link", "content": "escaped"})
    assert result.success is True
    assert outside.read_text() == "escaped"
