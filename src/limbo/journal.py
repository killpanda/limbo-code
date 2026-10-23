"""Session journal: the write side of a durable session file.

``sessions.py`` knows the file format (lines, records, load/merge rules).
This module owns *when and in what order* records reach the file — the
rules that keep the journal recoverable after a crash at any point:

- **watermark** — how many in-memory messages are already on disk, so a
  checkpoint appends only what is new;
- **full-rewrite fallback** — a failed or partial append (or a history
  reshaped in place) forces the next checkpoint to rebuild the file from
  memory, because re-appending would duplicate records;
- **what a rewrite drops** — ``save_session`` writes meta, compactions and
  messages only, so the live queue is re-journaled right after it, and
  dangling intents are never rewritten (a rewrite only happens at a
  checkpoint, when every journaled call has its result in history);
- **record order** — messages summarized before they reached disk go
  ahead of the compaction record that summarizes them; a ``dequeued``
  record goes after the message carrying the item (at-least-once: a crash
  between the two re-queues a message, never loses one);
- **meta** — a title change rewrites the meta line (listing reads only
  that line), any other change appends a ``meta_patch``.

The agent owns the state (messages, meta, queue) and calls the domain
verbs below; it never builds a record or tracks what is on disk. All
methods are thread-safe (checkpoints run in a worker thread while queue
changes come from the event loop); ``lock`` is re-entrant so the agent can
make a queue mutation and its journal call one atomic step.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from limbo.compaction import is_summary_message
from limbo.models import Message
from limbo.sessions import (
    COMPACTION_TYPE,
    META_PATCH_TYPE,
    QUEUE_TYPE,
    TOOL_INTENT_TYPE,
    CompactionRecord,
    QueuedMessageRecord,
    SessionMeta,
    ToolIntentRecord,
    append_session_messages,
    append_session_records,
    rewrite_meta_line,
    save_session,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _meta_snapshot(meta: SessionMeta) -> str:
    # updated_at moves on every save; compare real meta content only.
    return json.dumps(
        meta.model_dump(exclude={"path", "updated_at"}),
        ensure_ascii=False,
        sort_keys=True,
    )


def _meta_patch(meta: SessionMeta) -> dict[str, Any]:
    return {"type": META_PATCH_TYPE, **meta.model_dump(exclude={"path", "id"})}


def _queued(item: QueuedMessageRecord) -> dict[str, Any]:
    return {"type": QUEUE_TYPE, "op": "queued", **item.model_dump()}


class SessionJournal:
    """Append-mostly writer for one session file.

    A fresh journal has nothing on disk: the first :meth:`checkpoint`
    writes the whole file. After a resume, :meth:`mark_synced` records
    that the file already matches memory.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self._saved_count = 0
        self._saved_meta: str | None = None
        # The title on the file's meta line (listing reads only that line).
        self._saved_title: str | None = None
        self._rewrite_needed = False
        # Compaction records not yet appended, and the messages they
        # summarized before those reached disk.
        self._pending_compactions: list[CompactionRecord] = []
        self._pending_summarized: list[Message] = []
        # Queue items that left the queue since the last checkpoint.
        self._pending_dequeued: list[str] = []

    # -- state ----------------------------------------------------------------

    def mark_synced(
        self, message_count: int, meta: SessionMeta, *, rewrite: bool = False
    ) -> None:
        """The file holds ``message_count`` messages and ``meta`` (resume).

        ``rewrite`` says memory diverged from the file anyway (resume
        repaired dangling calls), so the next checkpoint rewrites it.
        """
        with self.lock:
            self._saved_count = message_count
            self._saved_meta = _meta_snapshot(meta)
            self._saved_title = meta.title
            self._rewrite_needed = rewrite

    def invalidate(self) -> None:
        """Messages already on disk changed in place: rewrite next time."""
        with self.lock:
            self._rewrite_needed = True

    @property
    def rewrite_pending(self) -> bool:
        return self._rewrite_needed

    # -- immediate appends ----------------------------------------------------

    def queued(self, item: QueuedMessageRecord) -> None:
        """Journal a newly queued message now.

        Before the first checkpoint (or while a rewrite is pending) there
        is nothing to append to: the next checkpoint journals the queue.
        A failed append raises after scheduling that rewrite.
        """
        with self.lock:
            if not self.path.exists() or self._rewrite_needed:
                return
            try:
                append_session_records(self.path, [_queued(item)])
            except OSError:
                self._rewrite_needed = True
                raise

    def dequeued(self, ids: Iterable[str]) -> None:
        """Items left the queue; journaled after the next checkpoint's messages."""
        with self.lock:
            self._pending_dequeued.extend(ids)

    def record_intents(self, calls: Sequence[tuple[str, str, str]]) -> None:
        """Journal ``(call_id, name, replay)`` calls before they run.

        One append per batch. A recovered call reads "may have partially
        run" when its intent is on disk and "did not run" otherwise.
        """
        if not calls:
            return
        now = _utc_now_iso()
        records = [
            {
                "type": TOOL_INTENT_TYPE,
                **ToolIntentRecord(
                    call_id=call_id, name=name, replay=replay, created_at=now
                ).model_dump(),
            }
            for call_id, name, replay in calls
        ]
        with self.lock:
            append_session_records(self.path, records)

    def patch_meta(self, meta: SessionMeta) -> None:
        """Persist a meta change now as a ``meta_patch`` (no-op if unchanged).

        Before the first checkpoint there is no file: that checkpoint
        writes the meta line.
        """
        snapshot = _meta_snapshot(meta)
        with self.lock:
            if not self.path.exists() or snapshot == self._saved_meta:
                return
            append_session_records(self.path, [_meta_patch(meta)])
            self._saved_meta = snapshot

    # -- compaction -----------------------------------------------------------

    def compacted(
        self,
        record: CompactionRecord,
        before: Sequence[Message],
        split: int,
        after_len: int,
    ) -> None:
        """History ``before`` was compacted at ``split`` into ``after_len`` messages.

        The new history is ``[system, summary, *before[split:]]``. The
        record is appended at the next checkpoint (non-destructive: the
        summarized messages stay in the file). When the keep boundary is
        an earlier summary — which lives only in memory on the append path
        — the record could not be resolved on load, so that one
        compaction falls back to a full rewrite.
        """
        with self.lock:
            if is_summary_message(before[split]):
                self._rewrite_needed = True
                return
            kept = len(before) - split
            # Messages summarized before they reached disk, written ahead
            # of the record so the journal still holds them.
            self._pending_summarized.extend(before[self._saved_count : split])
            # Kept messages not yet on disk are the tail of the new history.
            unsaved = min(len(before) - self._saved_count, kept)
            self._saved_count = after_len - max(unsaved, 0)
            self._pending_compactions.append(record)

    # -- checkpoint -----------------------------------------------------------

    def checkpoint(
        self,
        meta: SessionMeta,
        messages: Sequence[Message],
        compactions: Sequence[CompactionRecord],
        queue: Iterable[QueuedMessageRecord],
    ) -> None:
        """Bring the file up to date with memory.

        Appends what is new; rewrites the whole file on first save or
        when a rewrite is pending. Raises ``OSError`` on failure, after
        scheduling a full rewrite so the next checkpoint repairs the file.
        """
        snapshot = _meta_snapshot(meta)
        with self.lock:
            if self._rewrite_needed or not self.path.exists():
                self._rewrite(meta, messages, compactions, queue, snapshot)
                return
            try:
                self._append(meta, messages, snapshot)
            except OSError:
                # Some records may already be on disk; re-appending them
                # would duplicate. Rebuild from memory next time.
                self._rewrite_needed = True
                raise
        # Keep the in-memory timestamp fresh (export headers read it).
        meta.updated_at = _utc_now_iso()

    def _rewrite(
        self,
        meta: SessionMeta,
        messages: Sequence[Message],
        compactions: Sequence[CompactionRecord],
        queue: Iterable[QueuedMessageRecord],
        snapshot: str,
    ) -> None:
        save_session(self.path, meta, list(messages), compactions)
        self._saved_count = len(messages)
        self._saved_meta = snapshot
        self._saved_title = meta.title
        self._rewrite_needed = False
        self._pending_compactions.clear()
        self._pending_summarized.clear()
        # The rewrite dropped the queue journal: re-journal what is still
        # queued (everything else has left the queue).
        self._pending_dequeued.clear()
        queued = [_queued(item) for item in queue]
        if queued:
            try:
                append_session_records(self.path, queued)
            except OSError:
                self._rewrite_needed = True
                raise

    def _append(
        self, meta: SessionMeta, messages: Sequence[Message], snapshot: str
    ) -> None:
        if meta.title != self._saved_title:
            rewrite_meta_line(self.path, meta)
            self._saved_title = meta.title
            self._saved_meta = snapshot
        elif snapshot != self._saved_meta:
            append_session_records(self.path, [_meta_patch(meta)])
            self._saved_meta = snapshot
        if self._pending_summarized:
            append_session_messages(self.path, self._pending_summarized)
            self._pending_summarized.clear()
        if self._pending_compactions:
            append_session_records(
                self.path,
                [
                    {"type": COMPACTION_TYPE, **c.model_dump()}
                    for c in self._pending_compactions
                ],
            )
            self._pending_compactions.clear()
        new_messages = list(messages[self._saved_count :])
        if new_messages:
            append_session_messages(self.path, new_messages)
            self._saved_count = len(messages)
        if self._pending_dequeued:
            append_session_records(
                self.path,
                [{"type": QUEUE_TYPE, "op": "dequeued", "ids": self._pending_dequeued}],
            )
            self._pending_dequeued = []
