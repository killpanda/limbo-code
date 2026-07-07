import tempfile
from pathlib import Path

import pytest

from limbo.tools.ls import LsTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "src").mkdir()
        (Path(tmp) / "README.md").write_text("# Hi")
        (Path(tmp) / ".env").write_text("x")
        yield Path(tmp)


def test_ls_lists_files_and_dirs(workdir):
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": "."})
    assert result.success is True
    assert "src/" in result.output
    assert "README.md" in result.output
    assert ".env" in result.output


def test_ls_limit(workdir):
    tool = LsTool(workdir=workdir)
    result = tool.execute({"path": ".", "limit": 2})
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2


def test_ls_rejects_paths_outside_workdir(workdir):
    tool = LsTool(workdir=workdir)
    assert tool.execute({"path": ".."}).success is False
    assert tool.execute({"path": "../"}).success is False
    assert tool.execute({"path": "/tmp"}).success is False
