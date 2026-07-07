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
