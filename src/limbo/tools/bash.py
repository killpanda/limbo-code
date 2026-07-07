"""Execute bash commands tool."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool

DEFAULT_TIMEOUT = 30.0


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        "WARNING: bash is not sandboxed and can access files outside the workdir."
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

    def execute(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)

        if not command:
            return ToolResult(success=False, error="No command provided.")

        if is_dangerous(command, self.dangerous_patterns):
            return ToolResult(
                success=False,
                error=f"Command blocked by safety policy: {command}",
            )

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Will execute on approval:\n$ {command}",
                requires_confirmation=True,
            )

        shell = shutil.which("bash") or "/bin/bash"

        try:
            proc = subprocess.run(
                [shell, "-c", command],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=float(timeout),
                cwd=str(self.workdir),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {timeout}s.",
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"Execution failed: {e}")

        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + f"[stderr]\n{proc.stderr}"

        if proc.returncode != 0:
            return ToolResult(
                success=False,
                output=output or None,
                error=f"Command failed with exit code {proc.returncode}.",
            )

        return ToolResult(success=True, output=output or "")


def is_dangerous(command: str, patterns: list[str]) -> bool:
    """Return True if command matches a dangerous pattern.

    .. warning::
        This check is heuristic only. It tokenizes the top-level command, so
        subshells (``bash -c ...``), command substitution (``$(rm ...)``),
        variable indirection, and other shell constructs can bypass it. Review
        all commands before confirming destructive actions.
    """
    tokens = shlex.split(command)
    for pattern in patterns:
        if not pattern:
            continue
        pattern_tokens = shlex.split(pattern)
        if len(pattern_tokens) == 1:
            name = pattern_tokens[0]
            for token in tokens:
                if token == name or Path(token).name == name:
                    return True
        elif tokens[: len(pattern_tokens)] == pattern_tokens:
            return True
    return False
