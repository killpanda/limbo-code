"""Surgical file edit tool."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool
from limbo.tools.mutation_queue import mutation_lock_for

BOM = "\ufeff"


def _strip_bom(content: str) -> tuple[str, str]:
    """Split a leading UTF-8 BOM from ``content`` (LLMs never see it)."""
    if content.startswith(BOM):
        return BOM, content[len(BOM):]
    return "", content


def _detect_line_ending(content: str) -> str:
    """The file's dominant line ending, inferred from the first newline."""
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    return "\r\n" if lf_idx > 0 and content[lf_idx - 1] == "\r" else "\n"


def _normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class _MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


def _apply_edits(content: str, edits: list[dict[str, str]], path: str) -> str:
    """Apply every edit against the SAME original content, pi-style.

    All ``old_text`` values are matched against the unmodified file (never
    incrementally against earlier replacements), must each be unique, and
    must not overlap. Replacements are then applied in reverse offset
    order so earlier match offsets stay valid.
    """
    total = len(edits)
    matched: list[_MatchedEdit] = []
    for i, edit in enumerate(edits):
        old_text = edit["old_text"]
        if old_text == "":
            if total == 1:
                raise _EditError("old_text cannot be empty.")
            raise _EditError(f"edits[{i}].old_text cannot be empty.")
        occurrences = content.count(old_text)
        if occurrences == 0:
            if total == 1:
                raise _EditError(f"old_text not found in {path}.")
            raise _EditError(
                f"Could not find edits[{i}] in {path}. The old_text must match "
                "exactly including all whitespace and newlines."
            )
        if occurrences > 1:
            if total == 1:
                raise _EditError(f"Text must be unique in {path}.")
            raise _EditError(
                f"Found {occurrences} occurrences of edits[{i}] in {path}. "
                "Each old_text must be unique. Please provide more context "
                "to make it unique."
            )
        matched.append(
            _MatchedEdit(
                edit_index=i,
                match_index=content.index(old_text),
                match_length=len(old_text),
                new_text=edit["new_text"],
            )
        )

    matched.sort(key=lambda m: m.match_index)
    for prev, curr in zip(matched, matched[1:]):
        if prev.match_index + prev.match_length > curr.match_index:
            raise _EditError(
                f"edits[{prev.edit_index}] and edits[{curr.edit_index}] "
                f"overlap in {path}. Merge them into one edit or target "
                "disjoint regions."
            )

    new_content = content
    for m in reversed(matched):
        new_content = (
            new_content[: m.match_index]
            + m.new_text
            + new_content[m.match_index + m.match_length :]
        )
    if new_content == content:
        raise _EditError(f"No changes to {path}.")
    return new_content


class _EditError(Exception):
    """Validation failure with a model-actionable message."""


class EditTool(BaseTool):
    name = "edit"
    description = (
        "Edit a single file using exact text replacement. Every edits[].old_text "
        "must match a unique, non-overlapping region of the original file. "
        "If two changes affect the same block or nearby lines, merge them into "
        "one edit instead of emitting overlapping edits. Do not include large "
        "unchanged regions just to connect distant changes. "
        "The displayed diff may not clearly indicate trailing-newline changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "edits": {
                "type": "array",
                "description": (
                    "One or more targeted replacements. Each edit is matched "
                    "against the original file, not incrementally. Do not "
                    "include overlapping or nested edits."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_text": {
                            "type": "string",
                            "description": (
                                "Exact text for one targeted replacement. It "
                                "must be unique in the original file and must "
                                "not overlap with any other edits[].old_text "
                                "in the same call."
                            ),
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text for this targeted edit.",
                        },
                    },
                    "required": ["old_text", "new_text"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raw_path = arguments.get("path", "")
        try:
            edits = self._prepare_edits(arguments)
        except _EditError as e:
            return ToolResult(success=False, error=str(e))
        target = self.resolve_existing(raw_path, noun="File")

        # The whole read-verify-write sequence must be serialized per file:
        # a concurrent write to the same file must not interleave.
        with mutation_lock_for(target):
            try:
                # newline="": disable universal-newline translation so CRLF
                # files keep their \r\n through read-detect-write.
                with target.open("r", encoding="utf-8", newline="") as f:
                    raw_content = f.read()
            except OSError as e:
                return ToolResult(success=False, error=f"Could not read file: {e}")
            except UnicodeDecodeError as e:
                return ToolResult(success=False, error=f"File is not UTF-8 text: {e}")

            # Strip the BOM before matching (models never include invisible
            # BOM bytes in old_text) and restore it on write.
            bom, body = _strip_bom(raw_content)
            # Normalize CRLF/CR to LF for matching; restore the file's own
            # line ending on write so edits never flip a file's endings.
            line_ending = _detect_line_ending(body)
            content = _normalize_to_lf(body)
            normalized_edits = [
                {
                    "old_text": _normalize_to_lf(edit["old_text"]),
                    "new_text": _normalize_to_lf(edit["new_text"]),
                }
                for edit in edits
            ]

            try:
                new_content = _apply_edits(content, normalized_edits, raw_path)
            except _EditError as e:
                return ToolResult(success=False, error=str(e))

            diff = self._make_diff(content, new_content)

            if line_ending != "\n":
                new_content = new_content.replace("\n", line_ending)
            try:
                with target.open("w", encoding="utf-8", newline="") as f:
                    f.write(bom + new_content)
            except OSError as e:
                return ToolResult(success=False, error=f"Could not write file: {e}")

        edited = f"Edited {raw_path}."
        if len(edits) > 1:
            edited = f"Edited {raw_path} ({len(edits)} replacements)."
        return ToolResult(success=True, output=f"{edited}\n{diff}")

    @staticmethod
    def _prepare_edits(arguments: dict[str, Any]) -> list[dict[str, str]]:
        """Normalize the call into an edits list.

        Accepts the legacy single-edit shape (``old_text``/``new_text`` at
        the top level) and an ``edits`` value sent as a JSON string, folding
        both into the canonical ``edits`` array.
        """
        edits = arguments.get("edits")
        # Some models send edits as a JSON string instead of an array.
        if isinstance(edits, str):
            try:
                parsed = json.loads(edits)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                edits = parsed
        result: list[dict[str, str]] = []
        if isinstance(edits, list):
            for item in edits:
                if (
                    isinstance(item, dict)
                    and isinstance(item.get("old_text"), str)
                    and isinstance(item.get("new_text"), str)
                ):
                    result.append(
                        {"old_text": item["old_text"], "new_text": item["new_text"]}
                    )
        # Legacy single-edit shape still works.
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if isinstance(old_text, str) and isinstance(new_text, str):
            result.append({"old_text": old_text, "new_text": new_text})
        if not result:
            raise _EditError(
                "edits must contain at least one {old_text, new_text} replacement."
            )
        return result

    def _make_diff(self, old: str, new: str) -> str:
        return "".join(
            difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="\n")
        )
