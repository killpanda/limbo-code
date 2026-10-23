"""Persistent curated memory (daily mode; Hermes' MEMORY.md/USER.md design).

Two plain-text files under ``~/.limbo/memory/`` that the user can open and
edit by hand:

- ``USER.md``  — who the user is: role, preferences, style.
- ``NOTES.md`` — the agent's notes: stable facts, conventions, corrections.

Entries are separated by ``ENTRY_DELIMITER``. Each file has a character
cap, so memory stays small enough to ride in every prompt.

A session takes a frozen :class:`MemorySnapshot` when its agent is built:
the ``memory`` tool writes to disk at once, but the system prompt keeps
the snapshot, so the prompt prefix (and its cache) never changes mid-
session. New entries take effect in the next session.
"""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from limbo.tools.mutation_queue import mutation_lock_for

DEFAULT_MEMORY_DIR = Path.home() / ".limbo" / "memory"

ENTRY_DELIMITER = "\n§\n"
TARGETS: dict[str, str] = {"user": "USER.md", "notes": "NOTES.md"}
ACTIONS = ("add", "patch", "remove")


class MemoryStoreError(Exception):
    """A memory operation that cannot be applied (nothing was written)."""


def memory_dir() -> Path:
    """Where the memory files live (resolved per call; tests redirect it)."""
    return DEFAULT_MEMORY_DIR


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass(frozen=True)
class MemorySnapshot:
    """The memory a session was started with (rendered into its prompt)."""

    user: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def render(self) -> str | None:
        """The ``<memory>`` prompt block, or None when there is nothing to show."""
        sections = []
        if self.user:
            sections.append(
                "<user_profile>\n"
                + "\n".join(f"- {_xml_escape(e)}" for e in self.user)
                + "\n</user_profile>"
            )
        if self.notes:
            sections.append(
                "<agent_notes>\n"
                + "\n".join(f"- {_xml_escape(e)}" for e in self.notes)
                + "\n</agent_notes>"
            )
        if not sections:
            return None
        return (
            "\n\n<memory>\n"
            "Persistent memory from earlier sessions (a snapshot taken when this "
            "session started; facts the user states now take precedence):\n\n"
            + "\n\n".join(sections)
            + "\n</memory>\n"
        )


class MemoryStore:
    """Reads and atomically rewrites the memory files."""

    def __init__(self, directory: Path, limits: dict[str, int]):
        self.directory = directory
        self.limits = limits

    def path(self, target: str) -> Path:
        return self.directory / TARGETS[target]

    def entries(self, target: str) -> list[str]:
        try:
            text = self.path(target).read_text(encoding="utf-8")
        except OSError:
            return []
        return [e.strip() for e in text.split(ENTRY_DELIMITER.strip()) if e.strip()]

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            user=tuple(self.entries("user")), notes=tuple(self.entries("notes"))
        )

    @staticmethod
    def size(entries: list[str]) -> int:
        """Characters the entries take on disk (the cap is checked on this)."""
        return len(ENTRY_DELIMITER.join(entries))

    def apply(self, target: str, operations: list[dict[str, Any]]) -> list[str]:
        """Apply ``operations`` all-or-nothing; returns the new entries.

        The character cap is checked on the final result only, so one batch
        can free room (remove/patch) and add in the same call.
        """
        if target not in TARGETS:
            raise MemoryStoreError(
                f"Unknown target {target!r}; use one of: {', '.join(TARGETS)}"
            )
        if not operations:
            raise MemoryStoreError("No operations given")
        path = self.path(target)
        with mutation_lock_for(path):
            entries = self.entries(target)
            for i, op in enumerate(operations, 1):
                entries = _apply_one(entries, op, i if len(operations) > 1 else None)
            limit = self.limits.get(target, 0)
            if limit and self.size(entries) > limit:
                raise MemoryStoreError(
                    f"{TARGETS[target]} would be {self.size(entries)} characters; "
                    f"the limit is {limit}. Nothing was written. Reissue one "
                    "batch that removes or shortens stale entries and adds the "
                    "new one together. Current entries:\n"
                    + _numbered(self.entries(target))
                )
            self._write(path, entries)
        return entries

    def _write(self, path: Path, entries: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            # One step of undo for the user: the previous version.
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        text = ENTRY_DELIMITER.join(entries) + ("\n" if entries else "")
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _numbered(entries: list[str]) -> str:
    if not entries:
        return "(empty)"
    return "\n".join(f"{i}. {e}" for i, e in enumerate(entries, 1))


def _apply_one(entries: list[str], op: dict[str, Any], index: int | None) -> list[str]:
    where = f"operation {index}: " if index is not None else ""
    if not isinstance(op, dict):
        raise MemoryStoreError(f"{where}each operation must be an object")
    action = op.get("action")
    content = op.get("content")
    old_text = op.get("old_text")
    if action not in ACTIONS:
        raise MemoryStoreError(f"{where}action must be one of: {', '.join(ACTIONS)}")
    if action in ("add", "patch"):
        if not isinstance(content, str) or not content.strip():
            raise MemoryStoreError(f"{where}{action} needs a non-empty content")
        if ENTRY_DELIMITER.strip() in content:
            raise MemoryStoreError(
                f"{where}content must not contain the entry separator "
                f"{ENTRY_DELIMITER.strip()!r}"
            )
    if action == "add":
        assert isinstance(content, str)
        entry = content.strip()
        if entry in entries:
            return entries
        return [*entries, entry]
    if not isinstance(old_text, str) or not old_text.strip():
        raise MemoryStoreError(f"{where}{action} needs old_text (text from one entry)")
    matches = [i for i, e in enumerate(entries) if old_text in e]
    if not matches:
        raise MemoryStoreError(f"{where}no entry contains {old_text!r}")
    if len(matches) > 1:
        raise MemoryStoreError(
            f"{where}{len(matches)} entries contain {old_text!r}; quote more of the entry"
        )
    i = matches[0]
    new = list(entries)
    if action == "remove":
        del new[i]
        return new
    assert isinstance(content, str)
    if new[i].count(old_text) > 1:
        raise MemoryStoreError(
            f"{where}{old_text!r} occurs more than once in the entry; quote more of it"
        )
    new[i] = new[i].replace(old_text, content).strip()
    return new


# -- /memory (shared by both frontends) --------------------------------------

MEMORY_USAGE = "用法：/memory 查看记忆 · /memory edit [user|notes] 用编辑器打开"


def describe(store: MemoryStore | None, snapshot: MemorySnapshot | None) -> str:
    """What /memory prints: each file's entries and usage."""
    if store is None:
        return "记忆已关闭（[daily] memory = false）"
    blocks = []
    for target, label in (("user", "关于你"), ("notes", "笔记")):
        entries = store.entries(target)
        limit = store.limits.get(target, 0)
        head = f"{label} · {store.path(target)} · {MemoryStore.size(entries)}/{limit} 字符"
        body = "\n".join(f"  - {e}" for e in entries) if entries else "  （空）"
        blocks.append(f"{head}\n{body}")
    if snapshot is not None and snapshot != store.snapshot():
        blocks.append("本会话之后的改动从下一个会话起生效。")
    return "\n\n".join(blocks)


def parse_memory_command(arg: str) -> str | None:
    """``/memory`` argument → None (show) or the target to edit.

    Raises ValueError on anything else (the caller prints MEMORY_USAGE).
    """
    words = arg.split()
    if not words:
        return None
    if words[0] == "edit" and len(words) <= 2:
        target = words[1] if len(words) == 2 else "user"
        if target in TARGETS:
            return target
    raise ValueError(arg)


def editor_command(path: Path) -> list[str]:
    """$VISUAL / $EDITOR (default vi) opening ``path``."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    return [*shlex.split(editor), str(path)]
