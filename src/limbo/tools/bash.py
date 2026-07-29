"""Execute bash commands tool."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from limbo.config import DEFAULT_DANGEROUS_COMMANDS
from limbo.models import ToolResult
from limbo.tools.base import MAX_OUTPUT_BYTES, BaseTool, truncate_output

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        "WARNING: bash is not sandboxed and can access files outside the workdir. "
        "Commands matching dangerous patterns (e.g. rm, git reset --hard) are "
        "rejected outright and cannot be confirmed. The filter is heuristic only: "
        "subshells, command substitution, variable indirection, options before the command "
        "name, and variable assignments before the command name "
        "(e.g. 'bash -c rm -rf /', '$(rm ...)', 'git -C /foo reset --hard', "
        "or 'VAR=1 rm -rf /') can bypass it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (optional, default 30, maximum 300)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Path, dangerous_patterns: list[str] | None = None):
        super().__init__(workdir)
        self.dangerous_patterns = dangerous_patterns or list(
            DEFAULT_DANGEROUS_COMMANDS
        )

    def run(self, arguments: dict[str, Any], dry_run: bool = False) -> ToolResult:
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
            return self.confirm(f"Will execute on approval:\n$ {command}")

        # Cap the effective timeout so a huge value cannot block the worker
        # indefinitely. The cap is documented in the tool schema above.
        try:
            requested_timeout = float(timeout)
        except (TypeError, ValueError):
            return ToolResult(success=False, error=f"Invalid timeout: {timeout}")
        effective_timeout = min(requested_timeout, MAX_TIMEOUT)

        shell = shutil.which("bash") or "/bin/bash"

        try:
            proc = subprocess.run(
                [shell, "-c", command],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=effective_timeout,
                cwd=str(self.workdir),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Command timed out after {effective_timeout}s.",
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"Execution failed: {e}")

        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + f"[stderr]\n{proc.stderr}"

        output = truncate_output(
            output,
            suffix=(
                f"\n\n[output truncated: exceeded {MAX_OUTPUT_BYTES} byte limit]"
            ),
        )

        if proc.returncode != 0:
            exit_msg = f"Command failed with exit code {proc.returncode}."
            output = f"{exit_msg}\n{output}" if output else exit_msg
            return ToolResult(
                success=False,
                output=output,
                error=exit_msg,
            )

        return ToolResult(success=True, output=output or "")


_CONTROL_OPERATOR_RE = re.compile(r"(;|&&|&|\|\||\|)")


def _tokenize_command(command: str) -> list[str]:
    """Split a command into tokens, treating shell control operators as separate tokens."""
    raw_tokens = shlex.split(command)
    tokens: list[str] = []
    for raw in raw_tokens:
        for part in _CONTROL_OPERATOR_RE.split(raw):
            if part:
                tokens.append(part)
    return tokens


def is_dangerous(command: str, patterns: list[str]) -> bool:
    """Return True if command matches a dangerous pattern.

    Single-token patterns are matched only in command-name positions: the first
    token of the command or the first token after a shell control operator
    (``;``, ``&&``, ``&``, ``||``, ``|``). Multi-token patterns are matched
    against the leading tokens starting at each command position.

    .. warning::
        This check is heuristic only. It tokenizes the top-level command, so
        subshells (``bash -c ...``), command substitution (``$(rm ...)``),
        variable indirection, variable assignments before a command name
        (``VAR=1 rm -rf /``), and options between the command name and the
        matched tokens (``git -C /foo reset --hard``) can bypass it. Review all
        commands before confirming destructive actions.
    """
    tokens = _tokenize_command(command)
    control_operators = {";", "&&", "&", "||", "|"}
    command_starts = [0]
    for i in range(1, len(tokens)):
        if tokens[i - 1] in control_operators:
            command_starts.append(i)

    for pattern in patterns:
        if not pattern:
            continue
        pattern_tokens = _tokenize_command(pattern)
        for start in command_starts:
            if start >= len(tokens):
                continue
            if len(pattern_tokens) == 1:
                name = pattern_tokens[0]
                token = tokens[start]
                if token == name or Path(token).name == name:
                    return True
            else:
                first_pattern = pattern_tokens[0]
                first_token = tokens[start]
                first_matches = (
                    first_token == first_pattern
                    or Path(first_token).name == Path(first_pattern).name
                )
                if first_matches and tokens[
                    start + 1 : start + len(pattern_tokens)
                ] == pattern_tokens[1:]:
                    return True
    return False
