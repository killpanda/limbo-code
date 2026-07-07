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


def test_read_custom_sensitive_files(workdir):
    (workdir / "secret.txt").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir, sensitive_files=["secret.txt"])
    result = tool.execute({"path": "secret.txt"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_read_non_sensitive_file_not_in_custom_list(workdir):
    (workdir / ".env").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir, sensitive_files=["secret.txt"])
    result = tool.execute({"path": ".env"})
    assert result.success is True


def test_read_blocks_ssh_path(workdir):
    ssh_dir = workdir / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Host example")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": ".ssh/config"})
    assert result.success is False
    assert "sensitive" in result.error.lower()


def test_read_rejects_negative_offset(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": -1})
    assert result.success is False
    assert "positive" in result.error.lower()


def test_read_rejects_negative_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "limit": -1})
    assert result.success is False
    assert "positive" in result.error.lower()
