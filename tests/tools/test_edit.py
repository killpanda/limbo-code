import tempfile
from pathlib import Path

import pytest

from limbo.tools.edit import EditTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_edit_success(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
        dry_run=False,
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "x = 42\ny = 2\n"


def test_edit_requires_exact_match(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x=1", "new_text": "x=42"},
        dry_run=False,
    )
    assert result.success is False
    assert "not found" in result.error.lower()


def test_edit_non_unique(workdir):
    (workdir / "a.py").write_text("x = 1\nx = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
        dry_run=False,
    )
    assert result.success is False
    assert "unique" in result.error.lower()


def test_edit_outside_workdir(workdir):
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "../x.py", "old_text": "a", "new_text": "b"},
        dry_run=False,
    )
    assert result.success is False
    assert "outside" in result.error.lower()


def test_edit_no_change(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 1"},
        dry_run=False,
    )
    assert result.success is False
    assert "no changes" in result.error.lower()


def test_edit_rejects_empty_old_text(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "", "new_text": "x = 42"},
        dry_run=False,
    )
    assert result.success is False
    assert "old_text cannot be empty" in result.error.lower()


def test_edit_dry_run_does_not_modify(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
        dry_run=True,
    )
    assert result.success is True
    assert result.requires_confirmation is True
    assert "Proposed edit" in result.output
    assert "x = 42" in result.output
    assert (workdir / "a.py").read_text() == "x = 1\ny = 2\n"


def test_edit_handles_os_error(workdir):
    target = workdir / "a.py"
    target.write_text("x = 1\n")
    target.chmod(0o444)
    try:
        tool = EditTool(workdir=workdir)
        result = tool.execute(
            {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
            dry_run=False,
        )
        assert result.success is False
        assert "could not write file" in result.error.lower()
    finally:
        target.chmod(0o644)


def test_edit_handles_read_error(workdir):
    target = workdir / "a.py"
    target.write_text("x = 1\n")
    target.chmod(0o000)
    try:
        tool = EditTool(workdir=workdir)
        result = tool.execute(
            {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
            dry_run=False,
        )
        assert result.success is False
        assert "could not read file" in result.error.lower()
    finally:
        target.chmod(0o644)


def test_edit_rejects_symlink_escape(workdir):
    outside = workdir.parent / "outside_edit_target.py"
    outside.write_text("x = 1\n")
    link = workdir / "escape_link.py"
    link.symlink_to(outside)
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "escape_link.py", "old_text": "x = 1", "new_text": "x = 42"},
        dry_run=False,
    )
    assert result.success is False
    assert "outside" in result.error.lower()
    assert outside.read_text() == "x = 1\n"


def test_edit_broken_symlink_returns_invalid_path(workdir):
    link = workdir / "broken_link.py"
    try:
        link.symlink_to("does_not_exist")
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "broken_link.py", "old_text": "a", "new_text": "b"},
        dry_run=False,
    )
    assert result.success is False
    assert "invalid path" in result.error.lower()
