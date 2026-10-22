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
# else (or any output redirection) is blocked while planning. This is a
# best-effort guard, not a security boundary: shell verbs are open-ended
# and flags can hide writes (see _flag_writes_file for the ones we know).
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
# Flags that turn an otherwise read-only command into a writer, or into an
# arbitrary-command runner. Explicit list: the allowlist never claimed to
# be exhaustive, but these are the common footguns.
_GIT_WRITE_ARGS = ("--ext-diff", "--textconv", "--open-files-in-pager")
_SED_WRITE_COMMAND = re.compile(
    r"(?:^|[;\n}])\s*(?:[0-9$]+|/[^/]*/)?\s*[wWe](?:\s|$)"
)
# `s/…/…/w file` — the write lives in the substitution flags.
_SED_SUBST_WRITE = re.compile(r"(?:^|[;\n}])\s*s(.).*?\1.*?\1[^;}\n]*\bw\b")


def _flag_writes_file(head: str, words: list[str]) -> bool:
    """Whether an allowlisted command carries a flag that writes a file.

    Only the commands where such a flag exists are checked; unknown flags
    stay allowed (best-effort, see the allowlist comment).
    """
    args = words[1:]
    if head == "sort":
        # -o/--output writes the sort result; --compress-program runs it.
        return any(
            a in ("-o", "--output", "--compress-program")
            or a.startswith("--output=")
            or a.startswith("--compress-program=")
            or (a.startswith("-o") and not a.startswith("--"))
            for a in args
        )
    if head == "date":
        # -s/--set changes the system clock.
        return any(a in ("-s", "--set") or a.startswith("--set=") for a in args)
    if head == "xxd":
        # -r reverses a hex dump into a file when given an output path.
        return any(
            a == "-r" or (a.startswith("-") and not a.startswith("--") and "r" in a[1:])
            for a in args
        )
    if head == "sed":
        # -i is handled at the call site; a script can also write (w/W)
        # or execute (e) as a command.
        return any(
            _SED_WRITE_COMMAND.search(a) or _SED_SUBST_WRITE.search(a)
            for a in args
        )
    return False


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
        head = words[0]
        if _flag_writes_file(head, words):
            return False
        if head == "git":
            rest = words[1:]
            # Skip global options; -C/-c take a separate value.
            while rest and rest[0].startswith("-"):
                rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
            if not rest:
                return False
            if any(
                a.startswith("--output")
                or a in _GIT_WRITE_ARGS
                or a.startswith("--open-files-in-pager")
                or a == "-O"
                for a in rest
            ):
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
