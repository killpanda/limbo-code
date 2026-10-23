"""The ``memory`` tool (daily mode): curate USER.md / NOTES.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.memory import ACTIONS, TARGETS, MemoryStore, MemoryStoreError
from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError

_DESCRIPTION = (
    "Save durable facts to persistent memory that survives across sessions. "
    "Memory is injected into every future session, so keep entries compact and "
    "high-signal. Writes take effect from the next session.\n\n"
    "HOW: make all changes to one target in ONE call via 'operations' (each item: "
    "{action, content?, old_text?}). The batch applies atomically and the "
    "character limit is checked only on the final result, so one call can remove "
    "or patch stale entries to free room and add new ones.\n\n"
    "WHEN: only for facts that hold in every session regardless of task: who the "
    "user is, their preferences and style, stable environment facts, standing "
    "conventions, corrections the user made.\n\n"
    "SKIP: trivial or easily re-discovered facts, raw data, task progress, "
    "completed-work logs, temporary TODOs. Reusable procedures belong in a skill.\n\n"
    "TARGETS: 'user' = who the user is (name, role, preferences, style). "
    "'notes' = your notes (environment, conventions, lessons)."
)


class MemoryTool(BaseTool):
    name = "memory"
    description = _DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": list(TARGETS),
                "description": "'user' for the user profile, 'notes' for your notes.",
            },
            "operations": {
                "type": "array",
                "description": "Changes to apply to the target, in order, all-or-nothing.",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "content": {
                            "type": "string",
                            "description": (
                                "add: the new entry. patch: the text that replaces "
                                "old_text (the rest of the entry is kept)."
                            ),
                        },
                        "old_text": {
                            "type": "string",
                            "description": (
                                "patch/remove: text from exactly one entry. patch "
                                "replaces this span; remove deletes the whole entry."
                            ),
                        },
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target", "operations"],
    }

    def __init__(self, workdir: Path, store: MemoryStore):
        super().__init__(workdir)
        self.store = store

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        target = arguments.get("target")
        operations = arguments.get("operations")
        if not isinstance(target, str):
            raise ToolError(f"target must be one of: {', '.join(TARGETS)}")
        if not isinstance(operations, list):
            raise ToolError("operations must be a list")
        try:
            entries = self.store.apply(target, operations)
        except MemoryStoreError as e:
            raise ToolError(str(e)) from e
        except OSError as e:
            raise ToolError(f"Could not write memory: {e}") from e
        limit = self.store.limits.get(target, 0)
        used = MemoryStore.size(entries)
        cap = f"/{limit}" if limit else ""
        return ToolResult(
            success=True,
            output=(
                f"Updated {TARGETS[target]} ({len(entries)} entries, "
                f"{used}{cap} chars). Done; takes effect next session."
            ),
        )
