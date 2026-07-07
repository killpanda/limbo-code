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
    result = tool.execute({"path": "new.txt", "content": "hello"}, dry_run=False)
    assert result.success is True
    assert result.requires_confirmation is False
    assert (workdir / "new.txt").read_text() == "hello"


def test_write_overwrites_file(workdir):
    (workdir / "x.txt").write_text("old")
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "x.txt", "content": "new"}, dry_run=False)
    assert result.success is True
    assert (workdir / "x.txt").read_text() == "new"


def test_write_outside_workdir(workdir):
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "../x.txt", "content": "x"}, dry_run=False)
    assert result.success is False
    assert "outside" in result.error.lower()


def test_write_dry_run_does_not_create_file(workdir):
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "new.txt", "content": "hello"}, dry_run=True)
    assert result.success is True
    assert result.requires_confirmation is True
    assert "Will create/overwrite" in result.output
    assert not (workdir / "new.txt").exists()


def test_write_rejects_symlink_escape(workdir):
    outside = workdir.parent / "outside_write_target.txt"
    link = workdir / "escape_link"
    link.symlink_to(outside)
    tool = WriteTool(workdir=workdir)
    result = tool.execute(
        {"path": "escape_link", "content": "escaped"}, dry_run=False
    )
    assert result.success is False
    assert "outside" in result.error.lower()
    assert not outside.exists()
