"""Append-only JSONL trace log for analyzing complete agent runs.

The trace lives in a ``traces/`` subdirectory of the session directory
(``traces/<session>.trace.jsonl``) and records everything the message
history cannot reconstruct after the fact: full LLM request bodies,
token usage, tool execution timing, and errors/interruptions.

Full fidelity is the contract — every record (including the entire LLM
request body, which can be hundreds of KB to MBs on long sessions) is
persisted, never sampled or truncated. What IS optimized is the write
path: :meth:`TraceLogger.log` only enqueues the record (microseconds),
while JSON serialization and the file write+flush run on a dedicated
background writer thread, so a huge ``llm_request`` body never blocks
the asyncio event loop (which is also the UI thread). Ordering is
preserved (single FIFO consumer); ``flush()`` provides a synchronous
barrier for readers (export, turn-end save); ``close()`` drains and
joins the writer before releasing the handle. Tracing must never break
the agent: all write failures degrade to a warning.
"""

from __future__ import annotations

import json
import queue
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACE_SUFFIX = ".trace.jsonl"

# How long close() waits for the writer thread to drain the queue before
# giving up and closing the handle anyway (a stuck OS write must not hang
# session teardown / app exit).
_CLOSE_DRAIN_TIMEOUT = 10.0


def trace_path_for(session_file: Path) -> Path:
    """Return the trace file path belonging to a session file.

    Traces live in a ``traces/`` subdirectory so session listing/globbing
    (``*.jsonl``) never picks them up.
    """
    return session_file.parent / "traces" / f"{session_file.stem}{TRACE_SUFFIX}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceLogger:
    """Thread-safe, append-only JSONL event log (one record per line).

    Enqueue-fast / write-later: ``log()`` assembles the record dict and
    hands it to a single background writer thread (daemon) that owns all
    serialization and file I/O. Callers never block on disk. Use
    :meth:`flush` at points where a reader must see everything logged so
    far (e.g. before exporting, or at the turn-end save), and :meth:`close`
    to drain and release the file handle.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        file_existed = self.path.exists()
        self._file = self.path.open("a", encoding="utf-8")
        if not file_existed:
            self.path.chmod(0o600)
        self._queue: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._failed = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._writer,
            name="limbo-trace-writer",
            daemon=True,
        )
        self._thread.start()

    # -- writer thread --------------------------------------------------------

    def _writer(self) -> None:
        """Consume the FIFO queue: serialize + write + flush each record.

        Every record is flushed individually, preserving the original
        crash-safety guarantee (a crash loses at most the records still in
        the queue). The sentinel ``None`` (pushed by :meth:`close`) stops
        the loop after all earlier records were written.

        ``task_done`` runs AFTER the file write: :meth:`flush` blocks on
        ``queue.join()``, so it must only return once the record is really
        on disk (barrier semantics for export / turn-end save).
        """
        while True:
            record = self._queue.get()
            try:
                if record is None:
                    return
                try:
                    line = json.dumps(record, ensure_ascii=False, default=str)
                except (TypeError, ValueError) as e:
                    # Extremely defensive: `default=str` should catch
                    # everything.
                    line = json.dumps(
                        {
                            "ts": record.get("ts") if record else None,
                            "type": record.get("type") if record else "?",
                            "serialization_error": str(e),
                        },
                        ensure_ascii=False,
                    )
                try:
                    self._file.write(line + "\n")
                    self._file.flush()
                except (OSError, ValueError) as e:
                    # ValueError covers writes on a file handle closed by
                    # close() after its drain timeout — the writer must
                    # degrade to a warning, never die with a traceback.
                    if not self._failed:
                        self._failed = True
                        warnings.warn(f"Trace log write failed: {e}", stacklevel=2)
            finally:
                self._queue.task_done()

    # -- public API -----------------------------------------------------------

    def log(self, event_type: str, **fields: Any) -> None:
        """Queue one event record. Never raises, never blocks on disk."""
        if self._closed:
            return
        record = {"ts": _utc_now_iso(), "type": event_type, **fields}
        self._queue.put(record)

    def flush(self) -> None:
        """Block until every record logged so far is on disk.

        Safe to call from any thread (used inside ``asyncio.to_thread`` at
        the turn-end save, so the UI never waits on it). No-op after
        :meth:`close`.
        """
        if self._closed:
            return
        self._queue.join()

    def close(self) -> None:
        """Drain the queue, stop the writer, release the file handle.

        Idempotent: safe to call from Agent.close() at every replacement
        point even when the UI tears down several agents in sequence.
        Records logged after close are dropped silently (the handle is
        gone), same as before.
        """
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)  # sentinel: FIFO keeps earlier records first
        self._thread.join(timeout=_CLOSE_DRAIN_TIMEOUT)
        try:
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
