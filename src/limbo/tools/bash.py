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
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr."
    )
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
                error=f"Command failed with exit code {exit_code}.",
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
