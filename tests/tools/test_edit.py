import tempfile
from pathlib import Path

import pytest

from limbo.tools.edit import EditTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_edit_success(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "x = 42\ny = 2\n"


def test_edit_non_unique(workdir):
    (workdir / "a.py").write_text("x = 1\nx = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"},
    )
    assert result.success is False
    assert "unique" in result.error.lower()


# -- edits[] multi-replacement (pi parity) ------------------------------------


# -- BOM / CRLF normalization ---------------------------------------------------


