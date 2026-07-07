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


def test_grep_rejects_path_outside_workdir(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "foo", "path": ".."})
    assert result.success is False
    assert "outside" in result.error.lower()


def test_grep_rejects_absolute_path_outside_workdir(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "foo", "path": "/etc"})
    assert result.success is False
    assert "outside" in result.error.lower()


def test_grep_python_fallback_respects_gitignore(workdir):
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "def ignored"})
    assert result.success is True
    assert "ignored.py" not in result.output
    assert "a.py" not in result.output


def test_grep_python_fallback_searches_directory(workdir):
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "def foo"})
    assert result.success is True
    assert "a.py" in result.output


def test_grep_python_fallback_does_not_follow_symlinks_outside_workdir(workdir):
    outside = workdir.parent / "outside_secret.txt"
    outside.write_text("SECRET_OUTSIDE_WORKDIR=1\n")
    (workdir / "escape_link").symlink_to(outside)
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "SECRET_OUTSIDE_WORKDIR"})
    assert result.success is True
    assert "SECRET_OUTSIDE_WORKDIR" not in result.output


def test_grep_python_fallback_uses_glob(workdir):
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "def ", "glob": "*.py"})
    assert result.success is True
    assert "a.py" in result.output
    assert "b.py" in result.output

    result = tool.execute({"pattern": "def ", "glob": "a.py"})
    assert result.success is True
    assert "a.py" in result.output
    assert "b.py" not in result.output


def test_grep_python_fallback_rejects_context(workdir):
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "def foo", "context": 2})
    assert result.success is False
    assert "ripgrep" in result.error.lower()
