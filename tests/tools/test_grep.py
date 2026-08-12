import shutil
import tempfile
from pathlib import Path

import pytest

from limbo.tools.grep import GrepTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.py").write_text("def foo():\n    pass\n")
        (Path(tmp) / "b.py").write_text("def bar():\n    pass\n")
        (Path(tmp) / ".gitignore").write_text("ignored.py\n")
        (Path(tmp) / "ignored.py").write_text("def ignored():\n    pass\n")
        yield Path(tmp)


def test_grep_basic(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo"})
    assert result.success is True
    assert "a.py" in result.output
    assert "def foo" in result.output


def test_grep_respects_gitignore(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def ignored"})
    assert result.success is True
    assert "ignored.py" not in result.output


def test_grep_fixed_string(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def ", "fixed_string": True})
    assert result.success is True
    assert "a.py" in result.output


def test_grep_allows_path_outside_workdir(workdir, tmp_path):
    """No workdir fence: directories outside the workdir are searchable."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.py").write_text("def foo():\n    pass\n")
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo", "path": str(outside)})
    assert result.success is True
    assert "x.py" in result.output


def test_grep_requires_ripgrep(workdir, monkeypatch):
    """ripgrep is a hard dependency: no PATH rg means an actionable error."""
    monkeypatch.setattr(shutil, "which", lambda name: None)
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo"})
    assert result.success is False
    assert "ripgrep" in result.error.lower()


def test_grep_supports_context(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo", "context": 1})
    assert result.success is True
    assert "pass" in result.output


def test_grep_uses_glob(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def ", "glob": "a.py"})
    assert result.success is True
    assert "a.py" in result.output
    assert "b.py" not in result.output


def test_grep_limit_is_per_file(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def ", "limit": 1})
    assert result.success is True
    # rg --max-count is per file: both files match once.
    assert "a.py" in result.output
    assert "b.py" in result.output


def test_grep_includes_dotenv_files(workdir):
    """No sensitive-file list: .env is searchable like any other file."""
    (workdir / ".env").write_text("SECRET=topsecret\n")
    tool = GrepTool(workdir=workdir)
    # rg skips hidden files in directory searches, so target the file
    # directly — the point is that .env is no longer refused/excluded.
    result = tool.execute({"pattern": "SECRET", "path": ".env"})
    assert result.success is True
    assert "topsecret" in result.output


def test_grep_broken_symlink_search_path_returns_invalid_path(workdir):
    link = workdir / "broken_link"
    try:
        link.symlink_to("does_not_exist")
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "foo", "path": "broken_link"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_grep_ripgrep_uses_relative_paths(workdir):

    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo"})
    assert result.success is True
    assert "a.py" in result.output
    # The host path prefix should not leak into search results.
    assert str(workdir) not in result.output
