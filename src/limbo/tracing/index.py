"""Incremental byte-offset index over a trace JSONL file.

A trace can grow to hundreds of MB (every ``llm_request`` carries the full
request body), so the viewer never ships the file whole. The index scans
the file once, keeping for each record its byte offset plus a *summary*
(the record minus its bulky fields); the full record is re-read by offset
only when a detail panel asks for it.

:meth:`TraceIndex.refresh` resumes from the last scanned byte, so polling a
live session costs only the newly appended bytes. A trailing line without
its newline is still being written by the trace writer thread and is left
for the next refresh.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fields dropped from summaries: they can be huge and the tree view never
# needs them (the detail panel fetches the full record by offset).
_BULKY_FIELDS = ("body", "output", "content", "reasoning", "traceback", "arguments")

# Leading characters kept as a preview of a dropped text field.
PREVIEW_CHARS = 200


@dataclass(frozen=True)
class IndexedRecord:
    offset: int
    summary: dict[str, Any]


def _preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


def summarize(record: dict[str, Any]) -> dict[str, Any]:
    """The record without bulky fields, with short previews in their place."""
    summary: dict[str, Any] = {}
    for key, value in record.items():
        if key == "body":
            if isinstance(value, dict):
                if value.get("model") is not None:
                    summary["model"] = value["model"]
                messages = value.get("messages", value.get("input"))
                if isinstance(messages, list):
                    summary["message_count"] = len(messages)
            continue
        if key in _BULKY_FIELDS:
            if value:
                summary[f"{key}_preview"] = _preview(value)
            continue
        if key == "tool_calls" and isinstance(value, list):
            # llm_response tool calls carry full arguments since schema 2.
            value = [
                {k: v for k, v in call.items() if k != "arguments"}
                for call in value
                if isinstance(call, dict)
            ]
        summary[key] = value
    return summary


class TraceIndex:
    """Thread-safe index of one trace file (the HTTP server is threaded)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[IndexedRecord] = []
        self._offsets: set[int] = set()
        self._scanned = 0
        self._lock = threading.Lock()

    @property
    def records(self) -> list[IndexedRecord]:
        with self._lock:
            return list(self._records)

    def refresh(self) -> int:
        """Index records appended since the last call; returns how many."""
        with self._lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                return 0
            if size < self._scanned:
                # Rewritten or truncated underneath us: start over.
                self._records.clear()
                self._offsets.clear()
                self._scanned = 0
            if size == self._scanned:
                return 0
            try:
                with self.path.open("rb") as f:
                    f.seek(self._scanned)
                    chunk = f.read(size - self._scanned)
            except OSError:
                return 0
            added = 0
            pos = 0
            while True:
                newline = chunk.find(b"\n", pos)
                if newline == -1:
                    break  # partial last line: still being written
                line = chunk[pos:newline]
                offset = self._scanned + pos
                pos = newline + 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                self._records.append(IndexedRecord(offset, summarize(record)))
                self._offsets.add(offset)
                added += 1
            self._scanned += pos
            return added

    def read(self, offset: int) -> dict[str, Any] | None:
        """The full record at ``offset``; only indexed offsets are served."""
        with self._lock:
            if offset not in self._offsets:
                return None
        try:
            with self.path.open("rb") as f:
                f.seek(offset)
                line = f.readline()
            record = json.loads(line)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return record if isinstance(record, dict) else None

    def previous_request(self, offset: int) -> int | None:
        """Offset of the agent ``llm_request`` before the one at ``offset``.

        Compaction requests are skipped: they carry a different prompt, so
        diffing against them would show the whole history as new.
        """
        previous = None
        for item in self.records:
            if item.offset >= offset:
                break
            summary = item.summary
            if summary.get("type") == "llm_request" and summary.get("kind") != "compaction":
                previous = item.offset
        return previous
