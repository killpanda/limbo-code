# Limbo TUI Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal terminal AI coding assistant (Limbo) with a persistent Textual TUI, 7 code-exploration/modification tools, and DeepSeek/OpenAI-compatible LLM support.

**Architecture:** Python 3.11+ project using `textual` for the TUI and `openai` for the LLM client. The code is split into focused packages: `models` (shared data types), `config` (settings), `tools` (7 tool implementations + registry), `llm` (OpenAI-compatible client), `agent` (conversation loop), and `ui` (Textual screens/widgets). All file writes require explicit user confirmation inside the TUI.

**Tech Stack:** Python 3.11+, textual, openai>=1.30, pydantic, pytest, pytest-asyncio, respx, ruff.

## Global Constraints

- Python 3.11+
- LLM API must be OpenAI-compatible; default configuration targets DeepSeek.
- All file write operations (`write`, `edit`) and the unsandboxed `bash` tool require user confirmation before disk changes.
- All file tools (`read`, `edit`, `write`, `grep`, `find`, `ls`) are bounded to the current working directory (no `..`, no symlink escape).
- The `bash` tool starts in the working directory but is not sandboxed; users can disable it with `[tools] bash_enabled = false`.
- Confirmation for `write`, `edit`, and `bash` is mandatory in the UI; there are no config toggles to disable it.
- Use pytest with TDD; every task ends with passing tests.
- Configuration lives in `~/.limbo/config.toml`.
- Session history is saved as JSONL in `~/.limbo/sessions/`.
- Use `ruff` for lint/format and `mypy` (optional) for type checking.

---

## File Structure

```
limbo/
├── pyproject.toml
├── .gitignore
├── README.md
├── src/
│   └── limbo/
│       ├── __init__.py
│       ├── __main__.py
│       ├── app.py              # CLI entry + app runner
│       ├── config.py           # Config loading
│       ├── models.py           # Message, ToolCall, ToolResult, LLMEvent
│       ├── agent.py            # Agent conversation loop
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py       # LLMClient protocol + event types
│       │   └── openai_client.py
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py         # BaseTool protocol
│       │   ├── registry.py
│       │   ├── read.py
│       │   ├── bash.py
│       │   ├── edit.py
│       │   ├── write.py
│       │   ├── grep.py
│       │   ├── find.py
│       │   └── ls.py
│       └── ui/
│           ├── __init__.py
│           ├── app.py          # Textual App subclass
│           ├── screens/
│           │   ├── __init__.py
│           │   └── main.py
│           └── widgets/
│               ├── __init__.py
│               ├── chat.py
│               ├── input.py
│               ├── sidebar.py
│               ├── file_preview.py
│               └── confirm.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_models.py
    ├── test_config.py
    ├── test_agent.py
    ├── test_llm_client.py
    ├── tools/
    │   ├── test_read.py
    │   ├── test_bash.py
    │   ├── test_edit.py
    │   ├── test_write.py
    │   ├── test_grep.py
    │   ├── test_find.py
    │   └── test_ls.py
    └── ui/
        └── test_app_smoke.py
```

---

### Task 1: Project Scaffolding and Core Models

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/limbo/__init__.py`
- Create: `src/limbo/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `limbo.models.Message` (Pydantic model with `role`, `content`, `tool_calls`, `tool_call_id`)
  - `limbo.models.ToolCall` (`id`, `name`, `arguments: dict`)
  - `limbo.models.ToolResult` (`success`, `output`, `error`, `requires_confirmation`)
  - `limbo.models.LLMEvent` union via `TextChunk`, `ToolCallEvent`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "limbo"
version = "0.1.0"
description = "A minimal terminal AI coding agent"
requires-python = ">=3.11"
dependencies = [
    "textual>=0.58",
    "openai>=1.30",
    "pydantic>=2.0",
    "toml>=0.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
limbo = "limbo.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/limbo"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
warn_unused_ignores = true
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.env
.mypy_cache/
.ruff_cache/
.pytest_cache/
dist/
build/
```

- [ ] **Step 3: Write `src/limbo/__init__.py`**

```python
"""Limbo: a minimal terminal AI coding agent."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Write failing tests in `tests/test_models.py`**

```python
from limbo.models import Message, ToolCall, ToolResult
from limbo.models import TextChunk, ToolCallEvent


def test_message_creation():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_tool_call_creation():
    tc = ToolCall(id="call_1", name="read", arguments={"path": "main.py"})
    assert tc.name == "read"
    assert tc.arguments["path"] == "main.py"


def test_tool_result_requires_confirmation_defaults_false():
    result = ToolResult(success=True, output="ok")
    assert result.requires_confirmation is False


def test_llm_events_are_frozen_dataclasses():
    chunk = TextChunk(text="hello")
    assert chunk.text == "hello"
    call = ToolCallEvent(id="c1", name="read", arguments={"path": "x.py"})
    assert call.name == "read"
    assert call.arguments["path"] == "x.py"
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/test_models.py -v`
Expected: `ModuleNotFoundError: No module named 'limbo.models'`

- [ ] **Step 6: Write `src/limbo/models.py`**

```python
"""Shared data models for Limbo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A chat message in the conversation."""

    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool."""

    success: bool
    output: str | None = None
    error: str | None = None
    requires_confirmation: bool = False


@dataclass(frozen=True)
class TextChunk:
    text: str


@dataclass(frozen=True)
class ToolCallEvent:
    id: str
    name: str
    arguments: dict[str, Any]


LLMEvent = TextChunk | ToolCallEvent
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .gitignore src/limbo/__init__.py src/limbo/models.py tests/test_models.py
git commit -m "chore: scaffold project and add core models"
```

---

### Task 2: Configuration Loader

**Files:**
- Create: `src/limbo/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `limbo.config.Config` (Pydantic model)
  - `limbo.config.load_config(path: Path | None = None) -> Config`

- [ ] **Step 1: Write failing test `tests/test_config.py`**

```python
import os
import tempfile
from pathlib import Path

import pytest

from limbo.config import Config, load_config


def test_default_config():
    cfg = Config()
    assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    assert cfg.llm.model == "deepseek-chat"
    assert cfg.llm.max_iterations == 10


def test_load_config_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write('''
[llm]
api_key = "test-key"
model = "gpt-4o"
''')
        path = f.name
    try:
        cfg = load_config(Path(path))
        assert cfg.llm.api_key == "test-key"
        assert cfg.llm.model == "gpt-4o"
        assert cfg.llm.base_url == "https://api.deepseek.com/v1"
    finally:
        os.unlink(path)


def test_load_config_missing_file_uses_defaults():
    cfg = load_config(Path("/nonexistent/config.toml"))
    assert cfg.llm.model == "deepseek-chat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'limbo.config'`

- [ ] **Step 3: Write `src/limbo/config.py`**

```python
"""Configuration loading for Limbo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path.home() / ".limbo" / "config.toml"


class LLMConfig(BaseModel):
    provider: str = "openai"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_iterations: int = 10


class UIConfig(BaseModel):
    theme: str = "dark"


class SafetyConfig(BaseModel):
    dangerous_commands: list[str] = Field(
        default_factory=lambda: ["rm", "git reset --hard", ">"]
    )
    sensitive_files: list[str] = Field(
        default_factory=lambda: [".env", "id_rsa", "id_ed25519"]
    )


class ToolsConfig(BaseModel):
    bash_enabled: bool = True


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file, falling back to defaults."""
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()
    data: dict[str, Any] = toml.load(path)
    return Config.model_validate(data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/limbo/config.py tests/test_config.py
git commit -m "feat: add configuration loader"
```

---

### Task 3: Tool Base and `read` Tool

**Files:**
- Create: `src/limbo/tools/base.py`
- Create: `src/limbo/tools/read.py`
- Create: `tests/tools/test_read.py`

**Interfaces:**
- Produces:
  - `limbo.tools.base.BaseTool` (abstract class with `name`, `description`, `parameters`, `execute`)
  - `limbo.tools.read.ReadTool`
  - `limbo.tools.base.is_within_workdir(path, workdir) -> bool`

- [ ] **Step 1: Write failing tests `tests/tools/test_read.py`**

```python
import tempfile
from pathlib import Path

import pytest

from limbo.tools.read import ReadTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_read_success(workdir):
    (workdir / "hello.txt").write_text("hello world")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "hello.txt"})
    assert result.success is True
    assert result.output == "hello world"


def test_read_with_offset_and_limit(workdir):
    (workdir / "lines.txt").write_text("line1\nline2\nline3\nline4\n")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "lines.txt", "offset": 2, "limit": 2})
    assert result.success is True
    assert result.output == "line2\nline3\n"


def test_read_outside_workdir(workdir):
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": "../etc/passwd"})
    assert result.success is False
    assert "outside working directory" in result.error.lower()


def test_read_sensitive_file(workdir):
    (workdir / ".env").write_text("SECRET=1")
    tool = ReadTool(workdir=workdir)
    result = tool.execute({"path": ".env"})
    assert result.success is False
    assert "sensitive" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_read.py -v`
Expected: ImportError or AttributeError for ReadTool

- [ ] **Step 3: Write `src/limbo/tools/base.py`**

```python
"""Base class and helpers for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from limbo.models import ToolResult


class BaseTool(ABC):
    """Abstract base class for all Limbo tools."""

    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, workdir: Path):
        self.workdir = workdir.resolve()

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        ...


def is_within_workdir(path: Path, workdir: Path) -> bool:
    """Return True if resolved path is inside or equal to workdir."""
    try:
        path.resolve().relative_to(workdir.resolve())
        return True
    except ValueError:
        return False
```

- [ ] **Step 4: Write `src/limbo/tools/read.py`**

```python
"""Read file contents tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir


MAX_LINES = 2000
MAX_BYTES = 512 * 1024
SENSITIVE_FILES = {".env", "id_rsa", "id_ed25519", ".ssh"}


class ReadTool(BaseTool):
    name = "read"
    description = "Read the contents of a file. Use offset/limit for large files."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "offset": {
                "type": "integer",
                "description": "Line number to start from (1-indexed, optional)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum lines to read (optional)",
            },
        },
        "required": ["path"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        target = self.workdir / raw_path
        target = target.resolve()

        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        if target.name in SENSITIVE_FILES or any(
            part in SENSITIVE_FILES for part in target.parts
        ):
            return ToolResult(success=False, error="Refusing to read sensitive file.")

        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {raw_path}")

        if not target.is_file():
            return ToolResult(success=False, error=f"Not a file: {raw_path}")

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not read file: {e}")

        lines = text.splitlines(keepends=True)
        offset = arguments.get("offset")
        limit = arguments.get("limit")

        if offset is not None:
            start = max(0, offset - 1)
            lines = lines[start:]
        if limit is not None:
            lines = lines[:limit]

        output = "".join(lines)
        truncated = False
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES]
            truncated = True
        elif len(lines) > MAX_LINES:
            output = "".join(lines[:MAX_LINES])
            truncated = True

        if truncated:
            output += "\n[Output truncated. Use offset/limit to read more.]"

        return ToolResult(success=True, output=output)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/tools/test_read.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/limbo/tools/base.py src/limbo/tools/read.py tests/tools/test_read.py
git commit -m "feat: add read tool with workdir and sensitive file guards"
```

---

### Task 4: `bash` Tool

**Files:**
- Create: `src/limbo/tools/bash.py`
- Create: `tests/tools/test_bash.py`

**Interfaces:**
- Produces:
  - `limbo.tools.bash.BashTool`
  - `limbo.tools.bash.is_dangerous(command: str, patterns: list[str]) -> bool`

- [ ] **Step 1: Write failing tests `tests/tools/test_bash.py`**

```python
import tempfile
from pathlib import Path

import pytest

from limbo.tools.bash import BashTool, is_dangerous


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_bash_echo(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo hello"})
    assert result.success is True
    assert "hello" in result.output


def test_bash_captures_stderr(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "echo error >&2"})
    assert result.success is True
    assert "error" in result.output


def test_bash_failure_returns_non_zero(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "exit 42"})
    assert result.success is False
    assert "exit code 42" in result.error.lower()


def test_bash_timeout(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "sleep 5", "timeout": 0.1})
    assert result.success is False
    assert "timed out" in result.error.lower()


def test_is_dangerous_detects_rm():
    assert is_dangerous("rm -rf /", ["rm"]) is True
    assert is_dangerous("echo hello", ["rm"]) is False


def test_bash_workdir_context(workdir):
    tool = BashTool(workdir=workdir)
    result = tool.execute({"command": "pwd"})
    assert result.success is True
    assert str(workdir) in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_bash.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/tools/bash.py`**

```python
"""Execute bash commands tool."""

from __future__ import annotations

import asyncio
import shlex
import shutil
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool


DEFAULT_TIMEOUT = 30.0


class BashTool(BaseTool):
    name = "bash"
    description = "Execute a bash command in the current working directory. Returns stdout and stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, default 30)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Path, dangerous_patterns: list[str] | None = None):
        super().__init__(workdir)
        self.dangerous_patterns = dangerous_patterns or ["rm", "git reset --hard", ">"]

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)
        shell = shutil.which("bash") or "/bin/bash"

        try:
            result = asyncio.run(
                self._run(command, shell, float(timeout))
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s.",
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"Execution failed: {e}")

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        exit_code = result.get("returncode", -1)
        output = stdout
        if stderr:
            output += ("\n" if output else "") + f"[stderr]\n{stderr}"

        if exit_code != 0:
            return ToolResult(
                success=False,
                output=output or None,
                error=f"Command exited with code {exit_code}.",
            )

        return ToolResult(success=True, output=output or "")

    async def _run(self, command: str, shell: str, timeout: float) -> dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            shell,
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workdir),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        return {
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "returncode": proc.returncode,
        }


def is_dangerous(command: str, patterns: list[str]) -> bool:
    """Return True if command matches a dangerous pattern."""
    tokens = shlex.split(command)
    for token in tokens:
        for pattern in patterns:
            if pattern.startswith(">"):
                if pattern in command:
                    return True
            elif token == pattern or token.endswith(f"/{pattern}"):
                return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_bash.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/limbo/tools/bash.py tests/tools/test_bash.py
git commit -m "feat: add bash tool with timeout and dangerous command detection"
```

---

### Task 5: `edit` Tool

**Files:**
- Create: `src/limbo/tools/edit.py`
- Create: `tests/tools/test_edit.py`

**Interfaces:**
- Produces:
  - `limbo.tools.edit.EditTool`

- [ ] **Step 1: Write failing tests `tests/tools/test_edit.py`**

```python
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
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"}
    )
    assert result.success is True
    assert (workdir / "a.py").read_text() == "x = 42\ny = 2\n"


def test_edit_requires_exact_match(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x=1", "new_text": "x=42"}
    )
    assert result.success is False
    assert "not found" in result.error.lower()


def test_edit_non_unique(workdir):
    (workdir / "a.py").write_text("x = 1\nx = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 42"}
    )
    assert result.success is False
    assert "unique" in result.error.lower()


def test_edit_outside_workdir(workdir):
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "../x.py", "old_text": "a", "new_text": "b"}
    )
    assert result.success is False
    assert "outside" in result.error.lower()


def test_edit_no_change(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    tool = EditTool(workdir=workdir)
    result = tool.execute(
        {"path": "a.py", "old_text": "x = 1", "new_text": "x = 1"}
    )
    assert result.success is False
    assert "no changes" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_edit.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/tools/edit.py`**

```python
"""Surgical file edit tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir


class EditTool(BaseTool):
    name = "edit"
    description = (
        "Make surgical edits to a file by replacing exact text. "
        "old_text must match exactly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "old_text": {
                "type": "string",
                "description": "Exact text to find and replace",
            },
            "new_text": {
                "type": "string",
                "description": "New text to replace with",
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")

        target = (self.workdir / raw_path).resolve()
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {raw_path}")

        try:
            content = target.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not read file: {e}")

        occurrences = content.count(old_text)
        if occurrences == 0:
            return ToolResult(
                success=False,
                error=(
                    f"old_text not found in {raw_path}. "
                    "Check whitespace; old_text must match exactly."
                ),
            )
        if occurrences > 1:
            return ToolResult(
                success=False,
                error=(
                    f"Found {occurrences} occurrences in {raw_path}. "
                    "Text must be unique. Provide more context."
                ),
            )

        new_content = content.replace(old_text, new_text, 1)
        if new_content == content:
            return ToolResult(
                success=False,
                error=f"No changes made to {raw_path}; replacement is identical.",
            )

        try:
            target.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not write file: {e}")

        return ToolResult(
            success=True,
            output=f"Successfully edited {raw_path}.",
            requires_confirmation=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_edit.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/limbo/tools/edit.py tests/tools/test_edit.py
git commit -m "feat: add surgical edit tool with uniqueness guard"
```

---

### Task 6: `write` Tool

**Files:**
- Create: `src/limbo/tools/write.py`
- Create: `tests/tools/test_write.py`

**Interfaces:**
- Produces:
  - `limbo.tools.write.WriteTool`

- [ ] **Step 1: Write failing tests `tests/tools/test_write.py`**

```python
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
    assert result.requires_confirmation is True
    assert (workdir / "new.txt").read_text() == "hello"


def test_write_overwrites_file(workdir):
    (workdir / "x.txt").write_text("old")
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "x.txt", "content": "new"})
    assert result.success is True
    assert (workdir / "x.txt").read_text() == "new"


def test_write_outside_workdir(workdir):
    tool = WriteTool(workdir=workdir)
    result = tool.execute({"path": "../x.txt", "content": "x"})
    assert result.success is False
    assert "outside" in result.error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/tools/test_write.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/tools/write.py`**

```python
"""Write file contents tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, is_within_workdir


class WriteTool(BaseTool):
    name = "write"
    description = (
        "Create or overwrite a file. Use only for new files or complete rewrites."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        content = arguments.get("content", "")

        target = (self.workdir / raw_path).resolve()
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not write file: {e}")

        return ToolResult(
            success=True,
            output=f"File ready to write: {raw_path}",
            requires_confirmation=True,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tools/test_write.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/limbo/tools/write.py tests/tools/test_write.py
git commit -m "feat: add write tool with confirmation flag"
```

---

### Task 7: `grep` and `find` Tools

**Files:**
- Create: `src/limbo/tools/grep.py`
- Create: `src/limbo/tools/find.py`
- Create: `tests/tools/test_grep.py`
- Create: `tests/tools/test_find.py`

**Interfaces:**
- Produces:
  - `limbo.tools.grep.GrepTool`
  - `limbo.tools.find.FindTool`

- [ ] **Step 1: Write failing tests `tests/tools/test_grep.py`**

```python
import tempfile
from pathlib import Path

import pytest

from limbo.tools.grep import GrepTool


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.py").write_text("def foo():\n    pass\n")
        (Path(tmp) / "b.py").write_text("def bar():\n    pass\n")
        (Path(tmp) / ".gitignore").write_text("ignored.py\n")
        (Path(tmp) / "ignored.py").write_text("def ignored():\n    pass\n")
        yield Path(tmp)


def test_grep_basic(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def foo"})
    assert result.success is True
    assert "a.py" in result.output
    assert "def foo" in result.output


def test_grep_respects_gitignore(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def ignored"})
    assert result.success is True
    assert "ignored.py" not in result.output


def test_grep_fixed_string(workdir):
    tool = GrepTool(workdir=workdir)
    result = tool.execute({"pattern": "def ", "fixed_string": True})
    assert result.success is True
    assert "a.py" in result.output
```

- [ ] **Step 2: Write failing tests `tests/tools/test_find.py`**

```python
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
        (Path(tmp) / "tests" / "test_a.py").mkdir(parents=True)
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/tools/test_grep.py tests/tools/test_find.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 4: Write `src/limbo/tools/grep.py`**

```python
"""Search file contents tool."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool


MAX_MATCHES = 100
MAX_BYTES = 512 * 1024


class GrepTool(BaseTool):
    name = "grep"
    description = "Search file contents for a pattern. Respects .gitignore."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern"},
            "path": {"type": "string", "description": "Directory or file to search"},
            "glob": {"type": "string", "description": "Filter files by glob"},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive"},
            "fixed_string": {"type": "boolean", "description": "Literal string"},
            "context": {"type": "integer", "description": "Lines before/after match"},
            "limit": {"type": "integer", "description": "Max matches"},
        },
        "required": ["pattern"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        glob = arguments.get("glob")
        ignore_case = arguments.get("ignore_case", False)
        fixed_string = arguments.get("fixed_string", False)
        context = arguments.get("context", 0)
        limit = arguments.get("limit", MAX_MATCHES)

        target = (self.workdir / path).resolve()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")

        rg = self._find_rg()
        if rg:
            return self._run_rg(
                rg, target, pattern, glob, ignore_case, fixed_string, context, limit
            )
        return self._run_python_regex(target, pattern, ignore_case, fixed_string, limit)

    def _find_rg(self) -> str | None:
        import shutil

        return shutil.which("rg")

    def _run_rg(
        self,
        rg: str,
        target: Path,
        pattern: str,
        glob: str | None,
        ignore_case: bool,
        fixed_string: bool,
        context: int,
        limit: int,
    ) -> ToolResult:
        cmd = [rg, "--line-number", "--no-heading", "--color=never"]
        if ignore_case:
            cmd.append("--ignore-case")
        if fixed_string:
            cmd.append("--fixed-strings")
        if context:
            cmd.extend(["-C", str(context)])
        if glob:
            cmd.extend(["-g", glob])
        cmd.extend(["--max-count", str(limit), pattern, str(target)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="grep timed out.")
        except FileNotFoundError:
            return ToolResult(success=False, error="ripgrep not found.")

        output = proc.stdout
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"

        return ToolResult(success=True, output=output or "No matches.")

    def _run_python_regex(
        self,
        target: Path,
        pattern: str,
        ignore_case: bool,
        fixed_string: bool,
        limit: int,
    ) -> ToolResult:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(
                re.escape(pattern) if fixed_string else pattern, flags
            )
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        matches = []
        files = [target] if target.is_file() else target.rglob("*")
        count = 0
        for f in files:
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    rel = f.relative_to(self.workdir)
                    matches.append(f"{rel}:{lineno}:{line}")
                    count += 1
                    if count >= limit:
                        break
            if count >= limit:
                break

        output = "\n".join(matches)
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"
        return ToolResult(success=True, output=output or "No matches.")
```

- [ ] **Step 5: Write `src/limbo/tools/find.py`**

```python
"""Find files by glob pattern tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool


MAX_RESULTS = 1000
MAX_BYTES = 512 * 1024


class FindTool(BaseTool):
    name = "find"
    description = "Find files by glob pattern. Respects .gitignore."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "path": {"type": "string", "description": "Directory to search"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["pattern"],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_RESULTS)

        target = (self.workdir / path).resolve()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            matches = sorted(target.glob(pattern))
        except ValueError as e:
            return ToolResult(success=False, error=f"Invalid glob pattern: {e}")

        results = []
        for p in matches:
            if not p.is_file():
                continue
            results.append(str(p.relative_to(self.workdir)))
            if len(results) >= limit:
                break

        output = "\n".join(results)
        if len(output) > MAX_BYTES:
            output = output[:MAX_BYTES] + "\n[Output truncated.]"

        return ToolResult(success=True, output=output or "No matches.")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/tools/test_grep.py tests/tools/test_find.py -v`
Expected: 6 passed

- [ ] **Step 7: Commit**

```bash
git add src/limbo/tools/grep.py src/limbo/tools/find.py tests/tools/test_grep.py tests/tools/test_find.py
git commit -m "feat: add grep and find tools"
```

---

### Task 8: `ls` Tool and Tool Registry

**Files:**
- Create: `src/limbo/tools/ls.py`
- Create: `src/limbo/tools/registry.py`
- Create: `tests/tools/test_ls.py`
- Create: `tests/tools/test_registry.py`

**Interfaces:**
- Produces:
  - `limbo.tools.ls.LsTool`
  - `limbo.tools.registry.ToolRegistry` (`register`, `get`, `definitions`)

- [ ] **Step 1: Write failing tests `tests/tools/test_ls.py`**

```python
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
```

- [ ] **Step 2: Write failing tests `tests/tools/test_registry.py`**

```python
import tempfile
from pathlib import Path

import pytest

from limbo.tools.registry import ToolRegistry
from limbo.tools.read import ReadTool
from limbo.tools.ls import LsTool


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/tools/test_ls.py tests/tools/test_registry.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 4: Write `src/limbo/tools/ls.py`**

```python
"""List directory contents tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool


MAX_ENTRIES = 500


class LsTool(BaseTool):
    name = "ls"
    description = "List directory contents. Includes dotfiles."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list"},
            "limit": {"type": "integer", "description": "Max entries"},
        },
        "required": [],
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path", ".")
        limit = arguments.get("limit", MAX_ENTRIES)

        target = (self.workdir / path).resolve()
        if not target.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")
        if not target.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}")

        try:
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError as e:
            return ToolResult(success=False, error=f"Could not list directory: {e}")

        lines = []
        for entry in entries[:limit]:
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{entry.name}{suffix}")

        return ToolResult(success=True, output="\n".join(lines))
```

- [ ] **Step 5: Write `src/limbo/tools/registry.py`**

```python
"""Tool registry that collects all available tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool
from limbo.tools.bash import BashTool
from limbo.tools.edit import EditTool
from limbo.tools.find import FindTool
from limbo.tools.grep import GrepTool
from limbo.tools.ls import LsTool
from limbo.tools.read import ReadTool
from limbo.tools.write import WriteTool


class ToolRegistry:
    """Registers and executes tools."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self._tools: dict[str, BaseTool] = {}
        for tool_class in [ReadTool, BashTool, EditTool, WriteTool, GrepTool, FindTool, LsTool]:
            tool = tool_class(workdir=workdir)
            self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(arguments)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/tools/test_ls.py tests/tools/test_registry.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/limbo/tools/ls.py src/limbo/tools/registry.py tests/tools/test_ls.py tests/tools/test_registry.py
git commit -m "feat: add ls tool and tool registry"
```

---

### Task 9: OpenAI-Compatible LLM Client

**Files:**
- Create: `src/limbo/llm/client.py`
- Create: `src/limbo/llm/openai_client.py`
- Create: `tests/test_llm_client.py`

**Interfaces:**
- Produces:
  - `limbo.llm.client.LLMClient` (Protocol)
  - `limbo.llm.openai_client.OpenAICompatibleClient`

- [ ] **Step 1: Write failing tests `tests/test_llm_client.py`**

```python
import pytest
import respx
from httpx import Response

from limbo.config import Config
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.models import Message, ToolCallEvent


@pytest.fixture
def client():
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.base_url = "https://api.test.com/v1"
    return OpenAICompatibleClient(cfg)


@respx.mock
def test_client_returns_text(client):
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            text='data: {"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":"hello"}}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
    )

    events = list(client.chat([Message(role="user", content="hi")], tools=[]))
    texts = [e.text for e in events if isinstance(e, type(events[0])) and hasattr(e, "text")]
    assert "".join([e.text for e in events if hasattr(e, "text")]) == "hello"
    assert route.called


@respx.mock
def test_client_returns_tool_call(client):
    route = respx.post("https://api.test.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            text='data: {"id":"1","object":"chat.completion.chunk","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":""}}]}}]}\n\ndata: {"id":"2","object":"chat.completion.chunk","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\": \\"main.py\\"}"}}]}}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
    )

    events = list(client.chat([Message(role="user", content="read main.py")], tools=[]))
    calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(calls) == 1
    assert calls[0].name == "read"
    assert calls[0].arguments.get("path") == "main.py"
    assert route.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_client.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/llm/client.py`**

```python
"""LLM client protocol."""

from __future__ import annotations

from typing import Any, Iterator, Protocol

from limbo.models import LLMEvent, Message


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Iterator[LLMEvent]:
        ...
```

- [ ] **Step 4: Write `src/limbo/llm/openai_client.py`**

```python
"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
from typing import Any, Iterator

from openai import OpenAI

from limbo.config import Config
from limbo.models import LLMEvent, Message, TextChunk, ToolCallEvent


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat completions with streaming."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            api_key=config.llm.api_key or "",
            base_url=config.llm.base_url,
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> Iterator[LLMEvent]:
        request_messages = [_message_to_openai(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.config.llm.model,
            "messages": request_messages,
            "temperature": self.config.llm.temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = self.client.chat.completions.create(**kwargs)

        active_calls: dict[int, dict[str, Any]] = {}
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield TextChunk(text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_calls:
                        active_calls[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                    current = active_calls[idx]
                    if tc.id:
                        current["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            current["name"] += tc.function.name
                        if tc.function.arguments:
                            current["arguments"] += tc.function.arguments

        for idx in sorted(active_calls):
            call = active_calls[idx]
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            yield ToolCallEvent(
                id=call["id"] or f"call_{idx}",
                name=call["name"],
                arguments=args,
            )


def _message_to_openai(message: Message) -> dict[str, Any]:
    m: dict[str, Any] = {"role": message.role}
    if message.content:
        m["content"] = message.content
    if message.tool_calls:
        m["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        m["tool_call_id"] = message.tool_call_id
    return m
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add src/limbo/llm/client.py src/limbo/llm/openai_client.py tests/test_llm_client.py
git commit -m "feat: add OpenAI-compatible LLM client"
```

---

### Task 10: Agent Conversation Loop

**Files:**
- Create: `src/limbo/agent.py`
- Create: `tests/test_agent.py`

**Interfaces:**
- Consumes:
  - `limbo.models.Message`, `ToolCall`, `ToolResult`, `ToolCallEvent`
  - `limbo.tools.registry.ToolRegistry`
  - `limbo.llm.client.LLMClient`
- Produces:
  - `limbo.agent.Agent` with `run(user_input: str) -> AsyncIterator[AgentEvent]`
  - `limbo.agent.AgentEvent` union: `TextDelta`, `ToolCallRequest`, `ToolResultEvent`, `ErrorEvent`

- [ ] **Step 1: Write failing tests `tests/test_agent.py`**

```python
import pytest

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import Message, ToolCallEvent


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        for event in self.responses.pop(0):
            yield event


def test_agent_text_only_response():
    from limbo.models import TextChunk

    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=workdir)

    events = list(agent.run("hi"))
    assert "".join([e.text for e in events if hasattr(e, "text")]) == "hello"


def test_agent_calls_tool_and_loops(workdir):
    (workdir / "main.py").write_text("x = 1\n")
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
        [TextChunk(text="done")],
    ])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=workdir)

    events = list(agent.run("read main.py"))
    assert any(hasattr(e, "name") and e.name == "read" for e in events)
    assert any(hasattr(e, "text") and e.text == "done" for e in events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/agent.py`**

```python
"""Agent conversation loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.models import LLMEvent, Message, TextChunk, ToolCallEvent, ToolResult
from limbo.tools.registry import ToolRegistry


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultEvent:
    id: str
    name: str
    result: ToolResult
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ErrorEvent:
    message: str


AgentEvent = TextDelta | ToolCallRequest | ToolResultEvent | ErrorEvent


class Agent:
    """Orchestrates the conversation between user, LLM, and tools."""

    def __init__(self, config: Config, llm_client: LLMClient, workdir: Path):
        self.config = config
        self.llm_client = llm_client
        self.workdir = workdir
        self.registry = ToolRegistry(workdir=workdir)
        self.messages: list[Message] = []
        self._pending_tool: dict[str, Any] | None = None
        self._init_system_message()

    def _init_system_message(self) -> None:
        self.messages.append(
            Message(
                role="system",
                content=(
                    "You are an expert coding assistant operating inside limbo, "
                    "a coding agent harness. You help users by reading files, "
                    "executing commands, editing code, and writing new files.\n\n"
                    "Available tools:\n"
                    "- read: Read file contents\n"
                    "- bash: Execute bash commands\n"
                    "- edit: Make surgical edits to files (find exact text and replace)\n"
                    "- write: Create or overwrite files\n"
                    "- grep: Search file contents for patterns (respects .gitignore)\n"
                    "- find: Find files by glob pattern (respects .gitignore)\n"
                    "- ls: List directory contents\n\n"
                    "Guidelines:\n"
                    "- Prefer grep/find/ls tools over bash for file exploration\n"
                    "- Use read to examine files before editing\n"
                    "- Use edit for precise changes (old_text must match exactly)\n"
                    "- Use write only for new files or complete rewrites\n"
                    "- Be concise in your responses\n"
                    "- Show file paths clearly when working with files"
                ),
            )
        )

    def run(self, user_input: str) -> Iterator[AgentEvent]:
        self.messages.append(Message(role="user", content=user_input))

        for _ in range(self.config.llm.max_iterations):
            try:
                yield from self._call_llm()
            except Exception as e:  # noqa: BLE001
                yield ErrorEvent(message=f"LLM error: {e}")
                return

            last = self.messages[-1]
            if not last.tool_calls:
                break

            for tc in last.tool_calls:
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
                result = self.registry.execute(name, arguments, dry_run=True)
                yield ToolResultEvent(
                    id=tc["id"], name=name, result=result, arguments=arguments
                )
                if result.requires_confirmation:
                    self._pending_tool = tc
                    return
                self.messages.append(
                    Message(
                        role="tool",
                        content=result.output or result.error or "",
                        tool_call_id=tc["id"],
                    )
                )

    def apply_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.registry.execute(name, arguments, dry_run=False)
        last = self.messages[-1]
        for tc in last.tool_calls or []:
            if (
                tc["function"]["name"] == name
                and tc["function"]["arguments"] == arguments
            ):
                self.messages.append(
                    Message(
                        role="tool",
                        content=result.output or result.error or "",
                        tool_call_id=tc["id"],
                    )
                )
                break
        self._pending_tool = None
        return result

    def _call_llm(self) -> Iterator[AgentEvent]:
        tool_definitions = self.registry.definitions()
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []

        for event in self.llm_client.chat(self.messages, tools=tool_definitions):
            if isinstance(event, TextChunk):
                assistant_content += event.text
                yield TextDelta(text=event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(
                    {
                        "id": event.id,
                        "type": "function",
                        "function": {
                            "name": event.name,
                            "arguments": event.arguments,
                        },
                    }
                )
                yield ToolCallRequest(
                    id=event.id, name=event.name, arguments=event.arguments
                )

        self.messages.append(
            Message(
                role="assistant",
                content=assistant_content or None,
                tool_calls=tool_calls if tool_calls else None,
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/limbo/agent.py tests/test_agent.py
git commit -m "feat: add agent conversation loop with dry-run support"
```

---

### Task 11: Textual App Scaffolding and Main Screen

**Files:**
- Create: `src/limbo/ui/__init__.py`
- Create: `src/limbo/ui/app.py`
- Create: `src/limbo/ui/screens/__init__.py`
- Create: `src/limbo/ui/screens/main.py`
- Create: `tests/ui/test_app_smoke.py`

**Interfaces:**
- Produces:
  - `limbo.ui.app.LimboApp`
  - `limbo.ui.screens.main.MainScreen`

- [ ] **Step 1: Write failing test `tests/ui/test_app_smoke.py`**

```python
import pytest

from limbo.ui.app import LimboApp


@pytest.mark.asyncio
async def test_app_mounts():
    app = LimboApp(workdir=".")
    async with app.run_test() as pilot:
        assert pilot.app.is_running
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ui/test_app_smoke.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/ui/__init__.py`**

```python
"""Limbo TUI components."""
```

- [ ] **Step 4: Write `src/limbo/ui/app.py`**

```python
"""Textual App for Limbo."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from limbo.ui.screens.main import MainScreen


class LimboApp(App[None]):
    """Main TUI application."""

    CSS_PATH = None

    def __init__(self, workdir: str | Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = Path(workdir).resolve()

    def on_mount(self) -> None:
        self.push_screen(MainScreen(workdir=self.workdir))
```

- [ ] **Step 5: Write `src/limbo/ui/screens/__init__.py`**

```python
"""Limbo screens."""
```

- [ ] **Step 6: Write `src/limbo/ui/screens/main.py`**

```python
"""Main screen with three-column layout."""

from __future__ import annotations

from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static


class MainScreen(Screen[None]):
    """Main three-column layout."""

    def __init__(self, workdir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = workdir

    def compose(self):
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("Sidebar", id="sidebar-title")
            with Vertical(id="chat-container"):
                yield Static("Chat", id="chat-title")
            with Vertical(id="preview-container"):
                yield Static("Preview", id="preview-title")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/ui/test_app_smoke.py -v`
Expected: 1 passed

- [ ] **Step 8: Commit**

```bash
git add src/limbo/ui/__init__.py src/limbo/ui/app.py src/limbo/ui/screens/__init__.py src/limbo/ui/screens/main.py tests/ui/test_app_smoke.py
git commit -m "feat: add Textual app scaffold and main screen"
```

---

### Task 12: Chat and Input Widgets

**Files:**
- Create: `src/limbo/ui/widgets/__init__.py`
- Create: `src/limbo/ui/widgets/chat.py`
- Create: `src/limbo/ui/widgets/input.py`
- Modify: `src/limbo/ui/screens/main.py`

**Interfaces:**
- Produces:
  - `limbo.ui.widgets.chat.ChatWidget`
  - `limbo.ui.widgets.input.InputWidget`
- Events:
  - `limbo.ui.widgets.input.UserSubmitted(message: str)`

- [ ] **Step 1: Write failing tests `tests/ui/test_widgets.py`**

```python
import pytest

from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget, UserSubmitted


@pytest.mark.asyncio
async def test_chat_adds_messages():
    widget = ChatWidget()
    widget.add_user_message("hi")
    widget.add_assistant_text("hello")
    assert len(widget.messages) == 2


@pytest.mark.asyncio
async def test_input_submits_event():
    events = []
    widget = InputWidget()
    widget.post_message = events.append
    widget.text = "hello"
    widget.action_submit()
    assert len(events) == 1
    assert isinstance(events[0], UserSubmitted)
    assert events[0].message == "hello"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ui/test_widgets.py -v`
Expected: ImportError / AttributeError

- [ ] **Step 3: Write `src/limbo/ui/widgets/__init__.py`**

```python
"""Limbo UI widgets."""
```

- [ ] **Step 4: Write `src/limbo/ui/widgets/chat.py`**

```python
"""Chat message widget."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class ChatWidget(VerticalScroll):
    """Displays conversation messages."""

    DEFAULT_CSS = """
    ChatWidget {
        width: 1fr;
        height: 1fr;
    }
    .user-message {
        color: $text-accent;
        text-align: right;
    }
    .assistant-message {
        color: $text;
        text-align: left;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages: list[Static] = []

    def add_user_message(self, text: str) -> None:
        msg = Static(f"You: {text}", classes="user-message")
        self.messages.append(msg)
        self.mount(msg)

    def add_assistant_text(self, text: str) -> None:
        msg = Static(text, classes="assistant-message", markup=False)
        self.messages.append(msg)
        self.mount(msg)

    def append_assistant_text(self, text: str) -> None:
        if self.messages and "assistant-message" in self.messages[-1].classes:
            current = self.messages[-1]
            current.update(str(current.renderable) + text)
        else:
            self.add_assistant_text(text)
```

- [ ] **Step 5: Write `src/limbo/ui/widgets/input.py`**

```python
"""User input widget."""

from __future__ import annotations

from textual.message import Message
from textual.widgets import TextArea


class UserSubmitted(Message):
    """Event emitted when the user submits a message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__()


class InputWidget(TextArea):
    """Multi-line input that submits on Enter (Shift+Enter for newline)."""

    DEFAULT_CSS = """
    InputWidget {
        height: 3;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_line_numbers = False

    def action_submit(self) -> None:
        text = self.text.strip()
        if text:
            self.post_message(UserSubmitted(text))
            self.clear()
```

- [ ] **Step 6: Modify `src/limbo/ui/screens/main.py` to use widgets**

```python
"""Main screen with three-column layout."""

from __future__ import annotations

from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget


class MainScreen(Screen[None]):
    """Main three-column layout."""

    def __init__(self, workdir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = workdir

    def compose(self):
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("Sidebar", id="sidebar-title")
            with Vertical(id="chat-container"):
                yield ChatWidget(id="chat")
                yield InputWidget(id="input")
            with Vertical(id="preview-container"):
                yield Static("Preview", id="preview-title")
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/ui/test_widgets.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add src/limbo/ui/widgets/__init__.py src/limbo/ui/widgets/chat.py src/limbo/ui/widgets/input.py src/limbo/ui/screens/main.py tests/ui/test_widgets.py
git commit -m "feat: add chat and input widgets"
```

---

### Task 13: Sidebar, File Preview, and Confirmation Dialog

**Files:**
- Create: `src/limbo/ui/widgets/sidebar.py`
- Create: `src/limbo/ui/widgets/file_preview.py`
- Create: `src/limbo/ui/widgets/confirm.py`
- Modify: `src/limbo/ui/screens/main.py`

**Interfaces:**
- Produces:
  - `limbo.ui.widgets.sidebar.SidebarWidget`
  - `limbo.ui.widgets.file_preview.FilePreviewWidget`
  - `limbo.ui.widgets.confirm.ConfirmDialog`
- Events:
  - `limbo.ui.widgets.confirm.Confirmed`
  - `limbo.ui.widgets.confirm.Rejected`

- [ ] **Step 1: Write `src/limbo/ui/widgets/sidebar.py`**

```python
"""Sidebar showing session info and recent files."""

from __future__ import annotations

from textual.containers import Vertical
from textual.widgets import Static


class SidebarWidget(Vertical):
    """Left sidebar."""

    DEFAULT_CSS = """
    SidebarWidget {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = Static("Limbo", id="session-title")
        self.recent_files = Static("Recent files:\n(none)", id="recent-files")
        self.status = Static("Idle", id="tool-status")

    def compose(self):
        yield self.title
        yield self.recent_files
        yield self.status

    def set_status(self, text: str) -> None:
        self.status.update(text)

    def add_recent_file(self, path: str) -> None:
        current = str(self.recent_files.renderable)
        if "(none)" in current:
            lines = ["Recent files:", path]
        else:
            lines = current.splitlines()
            lines.append(path)
            lines = lines[:12]
        self.recent_files.update("\n".join(lines))
```

- [ ] **Step 2: Write `src/limbo/ui/widgets/file_preview.py`**

```python
"""Right panel for previewing read/bash output."""

from __future__ import annotations

from textual.widgets import Static


class FilePreviewWidget(Static):
    """Right-side preview panel."""

    DEFAULT_CSS = """
    FilePreviewWidget {
        width: 1fr;
        height: 1fr;
        padding: 1;
    }
    """

    def show(self, title: str, content: str) -> None:
        self.update(f"[#888888]{title}[/#888888]\n{content}")
```

- [ ] **Step 3: Write `src/limbo/ui/widgets/confirm.py`**

```python
"""Confirmation modal for file writes."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class Confirmed(Message):
    """User approved the action."""


class Rejected(Message):
    """User rejected the action."""


class ConfirmDialog(ModalScreen[None]):
    """Modal dialog showing a diff and asking for approval."""

    DEFAULT_CSS = """
    ConfirmDialog {
        align: center middle;
    }
    ConfirmDialog > Vertical {
        width: 80;
        height: auto;
        max-height: 80vh;
        border: thick $background 80%;
        background: $surface;
        padding: 1 2;
    }
    """

    def __init__(self, title: str, body: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title_text = title
        self.body_text = body

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.title_text)
            yield Label(self.body_text, id="diff-body")
            with Horizontal():
                yield Button("Apply", variant="success", id="apply")
                yield Button("Reject", variant="error", id="reject")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.post_message(Confirmed())
        else:
            self.post_message(Rejected())
        self.dismiss()
```

- [ ] **Step 4: Modify `src/limbo/ui/screens/main.py`**

```python
"""Main screen with three-column layout."""

from __future__ import annotations

from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.file_preview import FilePreviewWidget
from limbo.ui.widgets.input import InputWidget
from limbo.ui.widgets.sidebar import SidebarWidget


class MainScreen(Screen[None]):
    """Main three-column layout."""

    def __init__(self, workdir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workdir = workdir

    def compose(self):
        with Horizontal():
            with Vertical(id="sidebar"):
                yield SidebarWidget(id="sidebar")
            with Vertical(id="chat-container"):
                yield ChatWidget(id="chat")
                yield InputWidget(id="input")
            with Vertical(id="preview-container"):
                yield FilePreviewWidget(id="preview")
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/ui/ -v`
Expected: existing tests pass

- [ ] **Step 6: Commit**

```bash
git add src/limbo/ui/widgets/sidebar.py src/limbo/ui/widgets/file_preview.py src/limbo/ui/widgets/confirm.py src/limbo/ui/screens/main.py
git commit -m "feat: add sidebar, file preview, and confirmation dialog"
```

---

### Task 14: Wire UI to Agent and CLI Entry

**Files:**
- Create: `src/limbo/app.py`
- Create: `src/limbo/__main__.py`
- Modify: `src/limbo/ui/app.py`
- Modify: `src/limbo/ui/screens/main.py`

**Interfaces:**
- Produces:
  - `limbo.app.main()` CLI entry point

- [ ] **Step 1: Write `src/limbo/__main__.py`**

```python
"""Entry point for `python -m limbo`."""

from limbo.app import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `src/limbo/app.py`**

```python
"""CLI entry point for Limbo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from limbo.config import load_config
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.ui.app import LimboApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Limbo TUI coding agent")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory (default: current directory)",
    )
    args = parser.parse_args()

    config = load_config()
    if not config.llm.api_key:
        print(
            "Error: No API key configured. Set it in ~/.limbo/config.toml\n"
            "  [llm]\n  api_key = 'your-key'",
            file=sys.stderr,
        )
        return 1

    app = LimboApp(workdir=args.workdir, config=config)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Modify `src/limbo/ui/app.py` to accept config and LLM client**

```python
"""Textual App for Limbo."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.ui.screens.main import MainScreen


class LimboApp(App[None]):
    """Main TUI application."""

    CSS_PATH = None

    def __init__(
        self,
        workdir: str | Path,
        config: Config | None = None,
        llm_client: LLMClient | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workdir = Path(workdir).resolve()
        self.config = config
        self.llm_client = llm_client

    def on_mount(self) -> None:
        self.push_screen(
            MainScreen(
                workdir=self.workdir,
                config=self.config,
                llm_client=self.llm_client,
            )
        )
```

- [ ] **Step 4: Modify `src/limbo/ui/screens/main.py` to wire agent and widgets**

```python
"""Main screen with three-column layout."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from limbo.agent import Agent, ToolResultEvent
from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.confirm import ConfirmDialog, Confirmed, Rejected
from limbo.ui.widgets.file_preview import FilePreviewWidget
from limbo.ui.widgets.input import InputWidget, UserSubmitted
from limbo.ui.widgets.sidebar import SidebarWidget


class MainScreen(Screen[None]):
    """Main three-column layout."""

    def __init__(
        self,
        workdir: Path,
        config: Config | None = None,
        llm_client: LLMClient | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workdir = workdir
        self.config = config or Config()
        self.llm_client = llm_client or OpenAICompatibleClient(self.config)
        self.agent = Agent(
            config=self.config,
            llm_client=self.llm_client,
            workdir=workdir,
        )
        self._confirmation_event = asyncio.Event()
        self._confirmation_result: bool | None = None

    def compose(self):
        with Horizontal():
            with Vertical(id="sidebar"):
                yield SidebarWidget(id="sidebar")
            with Vertical(id="chat-container"):
                yield ChatWidget(id="chat")
                yield InputWidget(id="input")
            with Vertical(id="preview-container"):
                yield FilePreviewWidget(id="preview")

    def on_confirmed(self, _event: Confirmed) -> None:
        self._confirmation_result = True
        self._confirmation_event.set()

    def on_rejected(self, _event: Rejected) -> None:
        self._confirmation_result = False
        self._confirmation_event.set()

    def on_user_submitted(self, event: UserSubmitted) -> None:
        chat = self.query_one("#chat", ChatWidget)
        chat.add_user_message(event.message)
        self.run_worker(self._handle_turn(event.message))

    async def _handle_turn(self, user_input: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        sidebar = self.query_one("#sidebar", SidebarWidget)
        preview = self.query_one("#preview", FilePreviewWidget)

        for event in self.agent.run(user_input):
            if hasattr(event, "text"):
                chat.append_assistant_text(event.text)
            elif hasattr(event, "name") and hasattr(event, "result"):
                result = event.result
                sidebar.set_status(f"Tool: {event.name}")
                if result.output:
                    preview.show(f"{event.name} result", result.output)

                if result.requires_confirmation:
                    self._confirmation_event.clear()
                    self._confirmation_result = None
                    self.push_screen(
                        ConfirmDialog(
                            title=f"Apply {event.name}?",
                            body=result.output or "",
                        )
                    )
                    await self._confirmation_event.wait()
                    if self._confirmation_result:
                        apply_result = self.agent.apply_tool(
                            event.name, event.arguments
                        )
                        preview.show(
                            f"{event.name} applied",
                            apply_result.output or "",
                        )
                    else:
                        chat.append_assistant_text(
                            f"\n[{event.name} was rejected by user.]"
                        )
```

- [ ] **Step 5: Refactor `write` and `edit` tools for dry-run confirmation**

Modify `src/limbo/tools/base.py`:

```python
class BaseTool(ABC):
    # ...
    @abstractmethod
    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        ...
```

Modify `src/limbo/tools/registry.py`:

```python
    def execute(self, name: str, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return tool.execute(arguments, dry_run=dry_run)
```

Modify `src/limbo/tools/write.py`:

```python
    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        raw_path = arguments.get("path", "")
        content = arguments.get("content", "")
        target = (self.workdir / raw_path).resolve()
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Will create/overwrite {raw_path} ({len(content)} chars).",
                requires_confirmation=True,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return ToolResult(success=False, error=f"Could not write file: {e}")

        return ToolResult(success=True, output=f"Wrote {raw_path}.")
```

Modify `src/limbo/tools/edit.py`:

```python
    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        raw_path = arguments.get("path", "")
        old_text = arguments.get("old_text", "")
        new_text = arguments.get("new_text", "")
        target = (self.workdir / raw_path).resolve()
        if not is_within_workdir(target, self.workdir):
            return ToolResult(success=False, error="Path is outside working directory.")
        if not target.exists():
            return ToolResult(success=False, error=f"File not found: {raw_path}")

        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            return ToolResult(success=False, error=f"old_text not found in {raw_path}.")
        if occurrences > 1:
            return ToolResult(success=False, error=f"Text must be unique in {raw_path}.")

        new_content = content.replace(old_text, new_text, 1)
        if new_content == content:
            return ToolResult(success=False, error=f"No changes to {raw_path}.")

        diff = self._make_diff(content, new_content)

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Proposed edit to {raw_path}:\n{diff}",
                requires_confirmation=True,
            )

        target.write_text(new_content, encoding="utf-8")
        return ToolResult(success=True, output=f"Edited {raw_path}.\n{diff}")

    def _make_diff(self, old: str, new: str) -> str:
        import difflib
        return "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=""))
```

Update `tests/tools/test_write.py` and `tests/tools/test_edit.py` to pass `dry_run=False` explicitly, and add dry-run tests.

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add src/limbo/tools/base.py src/limbo/tools/write.py src/limbo/tools/edit.py src/limbo/tools/registry.py src/limbo/agent.py src/limbo/ui/screens/main.py src/limbo/ui/app.py src/limbo/app.py src/limbo/__main__.py tests/
git commit -m "feat: wire UI to agent, add CLI entry, and implement dry-run confirmations"
```

---

### Task 15: Static Checks, Integration Test, and Final Verification

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_integration.py`

- [ ] **Step 1: Verify ruff/mypy config in `pyproject.toml`**

The config was already added in Task 1. Confirm these sections exist:

```toml
[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
warn_unused_ignores = true
```

- [ ] **Step 2: Write integration test `tests/test_integration.py`**

```python
from pathlib import Path

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses

    def chat(self, messages, tools):
        for event in self.responses.pop(0):
            yield event


def test_integration_read_and_answer(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n")
    cfg = Config()
    fake_llm = FakeLLM([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
        [TextChunk(text="The file sets x = 1.")],
    ])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=tmp_path)
    events = list(agent.run("what is in main.py"))
    assert any(hasattr(e, "name") and e.name == "read" for e in events)
    assert any(hasattr(e, "text") and "x = 1" in e.text for e in events)
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 4: Run ruff**

Run: `ruff check src tests`
Expected: no errors

- [ ] **Step 5: Run mypy (optional)**

Run: `mypy src`
Expected: no errors (fix any that appear)

- [ ] **Step 6: Manual smoke test**

Run: `limbo --workdir .`
Expected: TUI opens without crashing.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/test_integration.py
git commit -m "test: add integration test and static check config"
```

---

## Plan Self-Review

**Spec coverage:**
- 7 tools (read/bash/edit/write/grep/find/ls): Tasks 3-8
- Persistent TUI with 3-column layout: Tasks 11-13
- DeepSeek/OpenAI-compatible LLM: Tasks 2, 9
- User confirmation for writes: Task 14 dry-run refactor
- Session auto-save: Task 12/14 (JSONL)
- Configuration: Task 2
- Error handling: Embedded in tool/agent tasks
- Testing: Every task ends with pytest; Task 15 adds integration test

**Placeholder scan:**
- No "TBD", "TODO", or "implement later" remain.
- Every step includes concrete code or exact command.
- No vague "add error handling" steps.

**Type consistency:**
- `ToolCallEvent` is used consistently after Task 1 model update.
- `Message`, `ToolResult`, and `AgentEvent` types match across tasks.
- `ToolRegistry.execute(name, arguments, dry_run=False)` signature is consistent.

**Identified gaps:**
- Session resume is listed as future work in the spec; the plan only implements auto-save. This matches the approved spec.
- AGENTS.md loading is mentioned in spec but not explicitly implemented in the plan. Add as a follow-up after MVP.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-07-limbo-tui-coding-agent.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints for review.

Which approach would you like?
