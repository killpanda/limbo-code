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
from collections.abc import Callable
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
# else (or any output redirection) is blocked while planning. This is a
# best-effort guard, not a security boundary: shell verbs are open-ended
# and flags can hide writes (see _ARG_CHECKS for the ones we know).
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
# Redirections that do not write a file: fd duplication (2>&1) and
# discarding to /dev/null. Stripped before looking for a real write.
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_HARMLESS_REDIRECT = re.compile(r"\d?>>?\s*/dev/null|\d?>&\d")
_WRITE_REDIRECT = re.compile(r"(?<![0-9&])>|[0-9]>(?!&)|&>")
# Per-command argument checks. A verb on the allowlist can still write
# (or run arbitrary commands) through a flag; each checker below says
# whether its command's arguments keep it read-only. Explicit lists: the
# allowlist never claimed to be exhaustive, but these are the common
# footguns.
_GIT_WRITE_ARGS = frozenset({"--ext-diff", "--textconv", "-O"})
_SED_WRITE_COMMAND = re.compile(
    r"(?:^|[;\n}])\s*(?:[0-9$]+|/[^/]*/)?\s*[wWe](?:\s|$)"
)
# `s/…/…/w file` — the write lives in the substitution flags.
_SED_SUBST_WRITE = re.compile(r"(?:^|[;\n}])\s*s(.).*?\1.*?\1[^;}\n]*\bw\b")


def _git_args_read_only(args: list[str]) -> bool:
    # Skip global options; -C/-c take a separate value.
    while args and args[0].startswith("-"):
        args = args[2:] if args[0] in ("-C", "-c") else args[1:]
    if not args:
        return False
    if any(
        a in _GIT_WRITE_ARGS
        or a.startswith("--output")
        or a.startswith("--open-files-in-pager")
        for a in args
    ):
        return False
    sub, rest = args[0], args[1:]
    if sub in _GIT_LIST_ONLY:
        return not any(
            not a.startswith("-") or a in _GIT_LIST_WRITE_FLAGS for a in rest
        )
    return sub in _READ_ONLY_GIT


def _find_args_read_only(args: list[str]) -> bool:
    return not any(a in _FIND_WRITE_ACTIONS for a in args)


def _sed_args_read_only(args: list[str]) -> bool:
    # -i edits in place; a script can also write (w/W) or execute (e).
    return not any(
        a.startswith("-i")
        or _SED_WRITE_COMMAND.search(a)
        or _SED_SUBST_WRITE.search(a)
        for a in args
    )


def _sort_args_read_only(args: list[str]) -> bool:
    # -o/--output writes the sort result; --compress-program runs one.
    return not any(
        a in ("--output", "--compress-program")
        or a.startswith("--output=")
        or a.startswith("--compress-program=")
        or (a.startswith("-o") and not a.startswith("--"))
        for a in args
    )


def _date_args_read_only(args: list[str]) -> bool:
    # -s/--set changes the system clock.
    return not any(a in ("-s", "--set") or a.startswith("--set=") for a in args)


def _xxd_args_read_only(args: list[str]) -> bool:
    # -r reverses a hex dump into a file when given an output path.
    return not any(
        a.startswith("-") and not a.startswith("--") and "r" in a[1:]
        for a in args
    )


_ARG_CHECKS: dict[str, Callable[[list[str]], bool]] = {
    "git": _git_args_read_only,
    "find": _find_args_read_only,
    "sed": _sed_args_read_only,
    "sort": _sort_args_read_only,
    "date": _date_args_read_only,
    "xxd": _xxd_args_read_only,
}


def is_read_only_command(command: str) -> bool:
    """Whether every segment of ``command`` is a known read-only command."""
    probe = _QUOTED.sub("", command)
    probe = _HARMLESS_REDIRECT.sub("", probe)
    if _WRITE_REDIRECT.search(probe):
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
        head, args = words[0], words[1:]
        check = _ARG_CHECKS.get(head)
        if check is not None:
            if not check(args):
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

    ``run_code`` is blocked outright: it executes arbitrary Python, so a
    program could write files or spawn processes without ever touching a
    tool (the hook chain only sees tool calls).
    """

    name = "plan"

    def __init__(self, is_read_only_tool: Any) -> None:
        # Callable[[str], bool]: the registry's replay policy ("safe" tools
        # are exactly the read-only ones).
        self._is_read_only_tool = is_read_only_tool

    def before_tool(self, name: str, arguments: dict[str, Any]) -> BeforeDecision:
        if name == "run_code":
            return Block(
                PLAN_MODE_BLOCK.format(
                    what="the run_code tool (it runs arbitrary Python)"
                )
            )
        if self._is_read_only_tool(name):
            return None
        if name == "bash":
            command = str(arguments.get("command", ""))
            if is_read_only_command(command):
                return None
            return Block(PLAN_MODE_BLOCK.format(what=f"bash `{command}`"))
        return Block(PLAN_MODE_BLOCK.format(what=f"the {name} tool"))
