import tempfile
from pathlib import Path

import pytest

from limbo.tools.registry import ToolRegistry


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_registry_definitions(workdir):
    reg = ToolRegistry(workdir=workdir)
    defs = reg.definitions()
    names = {d["function"]["name"] for d in defs}
    assert "read" in names
    assert "ls" in names


def test_registry_execute_read(workdir):
    (workdir / "x.txt").write_text("hi")
    reg = ToolRegistry(workdir=workdir)
    result = reg.execute("read", {"path": "x.txt"})
    assert result.success is True
    assert result.output == "hi"
