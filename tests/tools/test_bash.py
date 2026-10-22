import tempfile
import time
from pathlib import Path

import pytest

from limbo.tools.bash import BashTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_bash_echo(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo hello"})
    assert result.success is True
    assert "hello" in result.output


def test_bash_timeout_kills_process_tree(workdir):
    """A timed-out command's whole tree is killed (pi's killProcessTree).

    With a detached grandchild that keeps the pipe open, a plain
    ``proc.kill()`` would leave the reader threads blocked on the pipe
    and the call hanging; killing the process group tears it all down.
    """
    tool = BashTool(workdir=workdir)
    start = time.monotonic()
    result = tool.execute({"command": "sleep 60 & sleep 60", "timeout": 0.2})
    elapsed = time.monotonic() - start
    assert result.success is False
    assert "timed out" in result.error.lower()
    assert elapsed < 10


