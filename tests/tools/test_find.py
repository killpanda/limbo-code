import tempfile
from pathlib import Path

import pytest

from limbo.tools.find import FindTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "src").mkdir()
        (Path(tmp) / "src" / "a.py").write_text("x")
        (Path(tmp) / "src" / "b.py").write_text("y")
        (Path(tmp) / "tests").mkdir(parents=True)
        (Path(tmp) / "tests" / "test_a.py").write_text("z")
        yield Path(tmp)


def test_find_all_py(workdir):
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*.py"})
    assert result.success is True
    assert "src/a.py" in result.output
    assert "src/b.py" in result.output
    assert "tests/test_a.py" in result.output


def test_find_with_limit(workdir):
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*.py", "limit": 2})
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2


@pytest.fixture
def workdir_with_gitignore():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "src").mkdir()
        (Path(tmp) / "src" / "a.py").write_text("x")
        (Path(tmp) / ".gitignore").write_text("ignored.py\nignored_dir/\n")
        (Path(tmp) / "ignored.py").write_text("y")
        (Path(tmp) / "ignored_dir").mkdir()
        (Path(tmp) / "ignored_dir" / "x.py").write_text("z")
        yield Path(tmp)


def test_find_respects_gitignore(workdir_with_gitignore):
    tool = FindTool(workdir=workdir_with_gitignore)
    result = tool.execute({"pattern": "**/*.py"})
    assert result.success is True
    assert "src/a.py" in result.output
    assert "ignored.py" not in result.output
    assert "ignored_dir/x.py" not in result.output


def test_find_rejects_path_outside_workdir(workdir):
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "*.py", "path": ".."})
    assert result.success is False
    assert "outside" in result.error.lower()


def test_find_rejects_absolute_path_outside_workdir(workdir):
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "*.py", "path": "/etc"})
    assert result.success is False
    assert "outside" in result.error.lower()


def test_find_does_not_follow_symlink_to_outside_directory(workdir):
    """A symlinked directory inside the workdir must not expose outside files."""
    import os

    outside = workdir.parent / f"find_outside_{workdir.name}"
    outside.mkdir()
    (outside / "secret.py").write_text("secret")

    link = workdir / "outside_link"
    try:
        os.symlink(outside, link)
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    (workdir / "inside.py").write_text("inside")
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*.py"})
    assert result.success is True
    assert "inside.py" in result.output
    assert "secret.py" not in result.output


def test_find_broken_symlink_search_path_returns_invalid_path(workdir):
    link = workdir / "broken_link"
    try:
        link.symlink_to("does_not_exist")
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "*.py", "path": "broken_link"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_find_in_granted_root_outside_workdir(workdir, tmp_path):
    outside = tmp_path / "outside"
    (outside / "pkg").mkdir(parents=True)
    (outside / "pkg" / "mod.py").write_text("x")
    shared: set[Path] = {outside.resolve()}
    tool = FindTool(workdir=workdir, allowed_roots=shared)
    result = tool.execute({"pattern": "**/*.py", "path": str(outside)})
    assert result.success is True
    assert str((outside / "pkg" / "mod.py").resolve()) in result.output


def test_find_outside_workdir_without_grant_fails(workdir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*.py", "path": str(outside)})
    assert result.success is False
    assert "outside working directory" in result.error


def test_find_skips_sensitive_files(workdir):
    """Sensitive files are hidden from results (aligned with read/grep)."""
    (workdir / ".env").write_text("SECRET=1")
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*"})
    assert result.success is True
    assert ".env" not in result.output


def test_find_skips_sensitive_directory(workdir):
    ssh_dir = workdir / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_rsa").write_text("KEY")
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*"})
    assert result.success is True
    assert ".ssh" not in result.output
    assert "id_rsa" not in result.output


def test_find_refuses_sensitive_target(workdir):
    ssh_dir = workdir / ".ssh"
    ssh_dir.mkdir()
    tool = FindTool(workdir=workdir)
    result = tool.execute({"pattern": "**/*", "path": ".ssh"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_find_custom_sensitive_files(workdir):
    (workdir / "secret.txt").write_text("SECRET=1")
    tool = FindTool(workdir=workdir, sensitive_files=["secret.txt"])
    result = tool.execute({"pattern": "**/*"})
    assert result.success is True
    assert "secret.txt" not in result.output
