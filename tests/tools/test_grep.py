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


def test_grep_skips_sensitive_files(workdir):
    """Sensitive files are excluded from search results (aligned with read)."""
    (workdir / ".env").write_text("SECRET=topsecret\n")
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "SECRET"})
    assert result.success is True
    assert ".env" not in result.output
    assert "topsecret" not in result.output


def test_grep_python_fallback_skips_sensitive_files(workdir):
    (workdir / ".env").write_text("SECRET=topsecret\n")
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "SECRET"})
    assert result.success is True
    assert "topsecret" not in result.output


def test_grep_skips_sensitive_directory(workdir):
    ssh_dir = workdir / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Host secret-host\n")
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "secret-host"})
    assert result.success is True
    assert "secret-host" not in result.output


def test_grep_refuses_sensitive_target(workdir):
    (workdir / ".env").write_text("SECRET=topsecret\n")
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "SECRET", "path": ".env"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_grep_custom_sensitive_files(workdir):
    (workdir / "secret.txt").write_text("SECRET=topsecret\n")
    tool = GrepTool(workdir=workdir, sensitive_files=["secret.txt"])
    result = tool.execute({"pattern": "SECRET"})
    assert result.success is True
    assert "topsecret" not in result.output
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


def test_grep_python_fallback_limit_is_global(workdir):
    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "def ", "limit": 1})
    assert result.success is True
    # Only one match total should be returned, even though both files match.
    assert len(result.output.splitlines()) == 1


def test_grep_broken_symlink_search_path_returns_invalid_path(workdir):
    link = workdir / "broken_link"
    try:
        link.symlink_to("does_not_exist")
    except OSError:
        pytest.skip("Symlinks not supported on this platform")

    tool = GrepTool(workdir=workdir)
    tool._find_rg = lambda: None
    result = tool.execute({"pattern": "foo", "path": "broken_link"})
    assert result.success is False
    assert "invalid path" in result.error.lower()


def test_grep_ripgrep_uses_relative_paths(workdir):
    rg = GrepTool(workdir=workdir)._find_rg()
    if not rg:
        pytest.skip("ripgrep not installed")

    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo"})
    assert result.success is True
    assert "a.py" in result.output
    # The host path prefix should not leak into search results.
    assert str(workdir) not in result.output
