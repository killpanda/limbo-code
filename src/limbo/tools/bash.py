"""Execute bash commands tool."""

from __future__ import annotations

import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from limbo.config import DEFAULT_DANGEROUS_COMMANDS
from limbo.models import ToolResult
from limbo.tools.base import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    BaseTool,
    truncate_tail,
)

DEFAULT_TIMEOUT = 30.0
MAX_TIMEOUT = 300.0


def _write_full_output(output: str) -> Path | None:
    """Persist the full command output to a temp file; None on failure."""
    try:
        path = Path(tempfile.gettempdir()) / f"limbo-output-{secrets.token_hex(8)}.log"
        path.write_text(output, encoding="utf-8")
        return path
    except OSError:
        return None


class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to the last {DEFAULT_MAX_LINES} lines or "
        f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). If truncated, "
        "the full output is saved to a temp file and its path is shown. "
        "Note: bash is not sandboxed — it can access files outside the working "
        "directory, and the file tools' guardrails do not apply to it. "
        "Destructive commands are blocked; if a command you need is blocked, "
        "ask the user to run it manually."
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

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout", DEFAULT_TIMEOUT)

        if not command:
            return ToolResult(success=False, error="No command provided.")

        if is_dangerous(command, self.dangerous_patterns):
            return ToolResult(
                success=False,
                error=f"Command blocked by safety policy: {command}",
            )

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

        # Truncate to the tail (errors and final results live at the end).
        # When truncated, offload the full output to a temp file so the
        # model can retrieve it with read. Note: bash runs via
        # subprocess.run (non-streaming), so the full output is already in
        # memory and offloading is a post-hoc file write — unlike pi, which
        # spills to disk incrementally while streaming.
        truncation = truncate_tail(output)
        if truncation.truncated:
            full_path = _write_full_output(output)
            hint = (
                f"[Showing lines {truncation.start_line}-"
                f"{truncation.total_lines} of {truncation.total_lines}"
            )
            if full_path is not None:
                hint += f". Full output: {full_path}]"
            else:
                hint += "; full output could not be saved to a temp file]"
            output = f"{truncation.content}\n\n{hint}"

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
    """Split a command into tokens, treating shell control operators as separate tokens.

    Falls back to naive whitespace splitting when ``shlex`` cannot parse the
    command (e.g. unbalanced quotes): the filter is a best-effort heuristic
    and must never raise on malformed input.
    """
    try:
        raw_tokens = shlex.split(command)
    except ValueError:
        raw_tokens = command.split()
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

    .. note::
        This is a best-effort heuristic against accidental destruction, not
        a security boundary. It tokenizes the top-level command, so
        subshells, command substitution, variable indirection, and similar
        shell constructs can bypass it. Commands that ``shlex`` cannot
        parse fall back to whitespace tokenization instead of raising.
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
