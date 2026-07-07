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
