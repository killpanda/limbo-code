import tempfile
from pathlib import Path

import pytest

from limbo.tools.read import ReadTool


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
