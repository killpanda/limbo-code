import tempfile
from pathlib import Path

import pytest

from limbo.tools.registry import ToolRegistry


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.mark.asyncio
async def test_registry_execute_read(workdir):
    (workdir / "x.txt").write_text("hi")
    reg = ToolRegistry(workdir=workdir)
    result = await reg.execute("read", {"path": "x.txt"})
    assert result.success is True
    assert result.output == "hi"


