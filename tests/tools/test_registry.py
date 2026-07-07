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


def test_registry_wires_safety_config(workdir):
    cfg = Config()
    cfg.safety.dangerous_commands = ["danger"]
    cfg.safety.sensitive_files = ["secret.txt"]
    reg = ToolRegistry(workdir=workdir, config=cfg)

    bash = reg.get("bash")
    assert bash is not None
    assert bash.dangerous_patterns == ["danger"]

    read = reg.get("read")
    assert read is not None
    assert "secret.txt" in read.sensitive_files


def test_registry_defaults_without_config(workdir):
    reg = ToolRegistry(workdir=workdir)
    bash = reg.get("bash")
    assert bash is not None
    assert "rm" in bash.dangerous_patterns
    read = reg.get("read")
    assert read is not None
    assert ".env" in read.sensitive_files


def test_registry_bash_disabled(workdir):
    cfg = Config()
    cfg.tools.bash_enabled = False
    reg = ToolRegistry(workdir=workdir, config=cfg)
    assert reg.get("bash") is None
    assert "bash" not in {d["function"]["name"] for d in reg.definitions()}
