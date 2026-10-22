"""Tool hook chain: ``before_tool`` / ``after_tool`` around every call.

``ToolRegistry.execute`` runs each registered hook in order. A
``before_tool`` hook may block the call (``Block``), replace its arguments
(``Rewrite``) or let it through (``None``); raising blocks it too, so a
broken hook fails closed (pi parity). An ``after_tool`` hook may replace
the result. Code Mode sub-calls go through ``ToolRegistry.execute`` as
well, so hooks apply to them unchanged.

No hook is mounted by default — there are deliberately no safety fences.
The only built-in hook is plan mode (``/plan``), which keeps the agent
read-only while it explores and proposes a plan.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any

from limbo.models import ToolResult


@dataclass(frozen=True)
class Block:
    """Refuse the call; ``reason`` is returned to the model as the error."""

    reason: str


@dataclass(frozen=True)
class Rewrite:
    """Run the call with these arguments instead."""

    arguments: dict[str, Any]


BeforeDecision = Block | Rewrite | None


class ToolHook:
    """Base class for hooks; override either method (both default to no-op)."""

    name = "hook"

    def before_tool(self, name: str, arguments: dict[str, Any]) -> BeforeDecision:
        return None

    def after_tool(
        self, name: str, arguments: dict[str, Any], result: ToolResult
    ) -> ToolResult | None:
        return None


# Commands plan mode lets bash run: read-only inspection only. Anything
# else (or any output redirection) is blocked while planning.
_READ_ONLY_COMMANDS = frozenset(
    {
        "cat", "head", "tail", "less", "wc", "ls", "tree", "pwd", "file",
        "stat", "du", "df", "grep", "rg", "ag", "find", "fd", "which",
        "whereis", "type", "echo", "printf", "sort", "uniq", "cut", "tr",
        "diff", "cmp", "basename", "dirname", "realpath", "readlink", "date",
        "printenv", "uname", "whoami", "id", "jq", "column", "nl",
        "od", "xxd", "hexdump", "md5sum", "sha256sum", "true", "false",
    }
)
_READ_ONLY_GIT = frozenset(
    {
        "status", "log", "diff", "show", "blame", "rev-parse", "ls-files",
        "ls-tree", "describe", "shortlog", "grep",
    }
)
# Listing forms only: any positional argument would create/rename/delete.
_GIT_LIST_ONLY = frozenset({"branch", "tag", "remote"})
_GIT_LIST_WRITE_FLAGS = frozenset(
    {"-d", "-D", "-m", "-M", "-c", "-C", "-f", "--delete", "--move", "--copy",
     "--force", "-u", "--set-upstream-to", "--unset-upstream"}
)
# find actions that write or run commands.
_FIND_WRITE_ACTIONS = frozenset(
    {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0",
     "-fprintf", "-fls"}
)
# Segment separators; a redirection to a file is a write.
_SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;\n]")
_WRITE_REDIRECT = re.compile(r"(?<![0-9&])>|[0-9]>(?!&)|&>")


def is_read_only_command(command: str) -> bool:
    """Whether every segment of ``command`` is a known read-only command."""
    if _WRITE_REDIRECT.search(command.replace(">/dev/null", "").replace(
        "> /dev/null", ""
    )):
        return False
    if "$(" in command or "`" in command:
        return False
    for segment in _SEGMENT_SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        if not words:
            continue
        head = words[0]
        if head == "git":
            rest = words[1:]
            # Skip global options; -C/-c take a separate value.
            while rest and rest[0].startswith("-"):
                rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
            if not rest:
                return False
            sub, args = rest[0], rest[1:]
            if sub in _GIT_LIST_ONLY:
                if any(
                    not a.startswith("-") or a in _GIT_LIST_WRITE_FLAGS
                    for a in args
                ):
                    return False
            elif sub not in _READ_ONLY_GIT:
                return False
        elif head == "find":
            if any(w in _FIND_WRITE_ACTIONS for w in words[1:]):
                return False
        elif head == "sed":
            if any(w == "-i" or w.startswith("-i") for w in words[1:]):
                return False
        elif head not in _READ_ONLY_COMMANDS:
            return False
    return True


PLAN_MODE_INSTRUCTION = (
    "Plan mode is on: you are read-only. Explore with read/grep/find/ls and "
    "read-only bash commands; edit, write and state-changing commands are "
    "blocked. Finish by presenting a concrete, step-by-step plan. The user "
    "turns plan mode off (/plan off) before anything is changed."
)

PLAN_MODE_BLOCK = (
    "Plan mode is on (read-only): {what} is blocked. Keep exploring with "
    "read-only tools, then present the plan; the user leaves plan mode "
    "with /plan off before anything is changed."
)


class PlanModeHook(ToolHook):
    """Read-only mode: only crash-replay-safe tools and read-only bash run.

    ``run_code`` itself is allowed — its tool sub-calls pass through this
    same hook, so a program cannot reach a write tool either.
    """

    name = "plan"

    def __init__(self, is_read_only_tool: Any) -> None:
        # Callable[[str], bool]: the registry's replay policy ("safe" tools
        # are exactly the read-only ones).
        self._is_read_only_tool = is_read_only_tool

    def before_tool(self, name: str, arguments: dict[str, Any]) -> BeforeDecision:
        if name == "run_code" or self._is_read_only_tool(name):
            return None
        if name == "bash":
            command = str(arguments.get("command", ""))
            if is_read_only_command(command):
                return None
            return Block(PLAN_MODE_BLOCK.format(what=f"bash `{command}`"))
        return Block(PLAN_MODE_BLOCK.format(what=f"the {name} tool"))
