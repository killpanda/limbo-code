import tempfile
from pathlib import Path

import pytest

from limbo.tools.write import WriteTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_write_creates_file(workdir):
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "new.txt", "content": "hello"})
    assert result.success is True
    assert (workdir / "new.txt").read_text() == "hello"


