import tempfile
from pathlib import Path

import pytest

from limbo.config import Config
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


@pytest.mark.asyncio
async def test_registry_execute_read(workdir):
    (workdir / "x.txt").write_text("hi")
    reg = ToolRegistry(workdir=workdir)
    result = await reg.execute("read", {"path": "x.txt"})
    assert result.success is True
    assert result.output == "hi"


def test_registry_bash_disabled(workdir):
    cfg = Config()
    cfg.tools.bash_enabled = False
    reg = ToolRegistry(workdir=workdir, config=cfg)
    assert reg.get("bash") is None
    assert "bash" not in {d["function"]["name"] for d in reg.definitions()}
