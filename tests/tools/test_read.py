import tempfile
from pathlib import Path

import pytest

from limbo.tools.read import ReadTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_read_with_offset_and_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\nline3\nline4\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": 2, "limit": 2})
    assert result.success is True
    assert result.output == "line2\nline3\n"


# -- image attachments (pi parity) ----------------------------------------------

PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00" * 32
)
