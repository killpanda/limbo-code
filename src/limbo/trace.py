"""Append-only JSONL trace log for analyzing complete agent runs.

The trace lives in a ``traces/`` subdirectory of the session directory
(``traces/<session>.trace.jsonl``) and records everything the message
history cannot reconstruct after the fact: full LLM request bodies,
token usage, tool execution timing, and errors/interruptions.

Writes are appended and flushed immediately so a crash or interrupt loses
at most the in-flight event. Tracing must never break the agent: all write
failures degrade to a warning.
"""

from __future__ import annotations

import json
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_SUFFIX = ".trace.jsonl"


def trace_path_for(session_file: Path) -> Path:
    """Return the trace file path belonging to a session file.

    Traces live in a ``traces/`` subdirectory so session listing/globbing
    (``*.jsonl``) never picks them up.
    """
    return session_file.parent / "traces" / f"{session_file.stem}{TRACE_SUFFIX}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceLogger:
    """Thread-safe, append-only JSONL event log (one record per line)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        file_existed = self.path.exists()
        self._file = self.path.open("a", encoding="utf-8")
        if not file_existed:
            self.path.chmod(0o600)
        self._lock = threading.Lock()
        self._failed = False

    def log(self, event_type: str, **fields: Any) -> None:
        """Append one event record. Never raises."""
        record = {"ts": _utc_now_iso(), "type": event_type, **fields}
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            # Extremely defensive: `default=str` should catch everything.
            line = json.dumps(
                {
                    "ts": record["ts"],
                    "type": event_type,
                    "serialization_error": str(e),
                },
                ensure_ascii=False,
            )
        try:
            with self._lock:
                self._file.write(line + "\n")
                self._file.flush()
        except OSError as e:
            if not self._failed:
                self._failed = True
                warnings.warn(f"Trace log write failed: {e}", stacklevel=2)

    def close(self) -> None:
        try:
            with self._lock:
                self._file.close()
        except OSError:
            pass


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Read all trace records, skipping malformed lines."""
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        pass
    return records
