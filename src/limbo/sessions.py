"""Session storage: save, load, list, find, and export sessions.

Sessions are stored as JSONL files. The first line is a metadata record
(``{"type": "meta", ...}``) followed by one message per line. Files written
by older versions (no meta line) still load — the id falls back to the file
stem and the title is empty.

This module has no UI dependencies.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from limbo.compaction import is_summary_message, make_summary_message
from limbo.models import Message
from limbo.trace import read_trace

META_TYPE = "meta"
SNAPSHOT_TYPE = "messages_snapshot"
COMPACTION_TYPE = "compaction"


class SessionNotFoundError(Exception):
    """Raised when no session matches a lookup."""


class AmbiguousSessionError(Exception):
    """Raised when an id prefix matches multiple sessions."""

    def __init__(self, prefix: str, matches: list[str]):
        self.prefix = prefix
        self.matches = matches
        super().__init__(
            f"Session id prefix '{prefix}' is ambiguous, matches: "
            + ", ".join(matches)
        )


class SessionMeta(BaseModel):
    """Metadata describing a session (stored as the first JSONL line)."""

    id: str
    workdir: str = ""
    model: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    # Session-scoped granted roots beyond the workdir (implicit user
    # grants); restored on resume so the fence stays consistent.
    allowed_roots: list[str] = Field(default_factory=list)
    # Populated by list/load; not written to disk.
    path: Path | None = None


class CompactionRecord(BaseModel):
    """One compaction event, stored as a JSONL line in the session file.

    ``first_kept_entry_id`` marks the keep boundary (pi's entry model):
    on load, messages before that entry are dropped and the stored
    ``summary`` is re-inserted as the conversation summary. Because the
    session file is fully rewritten after each compaction, the boundary is
    normally a no-op — it is a defensive fallback and an audit trail.
    """

    id: str
    created_at: str = ""
    first_kept_entry_id: str = ""
    summary: str = ""
    tokens_before: int | None = None
    tokens_after_estimate: int | None = None
    trigger: str = "auto"  # "auto" | "manual"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def derive_title(messages: list[Message], max_length: int = 50) -> str:
    """Derive a session title from the first user message."""
    for msg in messages:
        if msg.role == "user" and msg.content:
            title = " ".join(msg.content.split())
            if len(title) > max_length:
                title = title[: max_length - 1].rstrip() + "…"
            return title
    return ""


def save_session(
    path: Path,
    meta: SessionMeta,
    messages: list[Message],
    compactions: Sequence[CompactionRecord] = (),
) -> None:
    """Atomically write the whole session (meta + compactions + messages)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    meta.updated_at = _utc_now_iso()
    meta_line = json.dumps(
        {"type": META_TYPE, **meta.model_dump(exclude={"path"})},
        ensure_ascii=False,
    )
    tmp_file = path.with_suffix(".tmp")
    file_existed = path.exists()
    with tmp_file.open("w", encoding="utf-8") as f:
        f.write(meta_line + "\n")
        for record in compactions:
            f.write(
                json.dumps(
                    {"type": COMPACTION_TYPE, **record.model_dump()},
                    ensure_ascii=False,
                )
                + "\n"
            )
        for msg in messages:
            f.write(msg.model_dump_json() + "\n")
    os.replace(tmp_file, path)
    if not file_existed:
        path.chmod(0o600)


def _parse_meta(raw: dict[str, Any], fallback_id: str, path: Path) -> SessionMeta:
    data = {k: v for k, v in raw.items() if k != "type"}
    data.setdefault("id", fallback_id)
    return SessionMeta(path=path, **data)


def _apply_compactions(
    messages: list[Message], compactions: list[CompactionRecord]
) -> list[Message]:
    """Reconstruct the post-compaction history from the latest record.

    Only the most recent record participates in reconstruction (older ones
    stay on disk for audit). Since the session file is rewritten after each
    compaction, the drop is normally a no-op; if the boundary id is missing
    (hand-edited file, partial write), the record is ignored with a warning
    — better uncompacted than data loss.
    """
    if not compactions:
        return messages
    latest = compactions[-1]
    boundary = next(
        (i for i, m in enumerate(messages) if m.id == latest.first_kept_entry_id),
        None,
    )
    if boundary is None:
        warnings.warn(
            f"Compaction boundary {latest.first_kept_entry_id!r} not found; "
            "loading full history.",
            stacklevel=2,
        )
        return messages
    head = messages[:1] if messages and messages[0].role == "system" else []
    kept = messages[boundary:]
    # The summary normally already sits in the file right after the system
    # message (saved post-compaction); re-insert it only if missing.
    dropped = messages[len(head) : boundary]
    summary_msg = next((m for m in dropped if is_summary_message(m)), None)
    if summary_msg is None:
        summary_msg = make_summary_message(latest.summary)
    return [*head, summary_msg, *kept]


def load_session(
    path: Path,
) -> tuple[SessionMeta, list[Message], list[CompactionRecord]]:
    """Load a session file, tolerating legacy files and malformed lines.

    Returns the meta, the reconstructed message history (with compaction
    applied), and the raw compaction records (the agent uses the latest one
    to restore its iterative-summary chain).
    """
    meta: SessionMeta | None = None
    messages: list[Message] = []
    compactions: list[CompactionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == META_TYPE:
                meta = _parse_meta(record, path.stem, path)
                continue
            if record.get("type") == COMPACTION_TYPE:
                data = {k: v for k, v in record.items() if k != "type"}
                try:
                    compactions.append(CompactionRecord(**data))
                except ValueError:
                    continue
                continue
            try:
                messages.append(Message(**record))
            except ValueError:
                continue
    if meta is None:
        # Legacy file without a meta line.
        meta = SessionMeta(id=path.stem, path=path)
    return meta, _apply_compactions(messages, compactions), compactions


def _read_meta_line(path: Path) -> SessionMeta:
    """Read only the first line of a session file (O(1) listing)."""
    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        first = ""
    if first:
        try:
            record = json.loads(first)
            if record.get("type") == META_TYPE:
                return _parse_meta(record, path.stem, path)
        except (json.JSONDecodeError, ValueError):
            pass
    # Legacy file: fall back to stem + mtime for sorting.
    mtime = path.stat().st_mtime
    updated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(
        timespec="seconds"
    )
    return SessionMeta(id=path.stem, updated_at=updated, path=path)


def list_sessions(
    session_dir: Path, workdir: Path | None = None
) -> list[SessionMeta]:
    """List sessions, most recently updated first.

    When ``workdir`` is given, only sessions recorded for that directory are
    returned.
    """
    if not session_dir.is_dir():
        return []
    wanted = str(workdir.resolve()) if workdir is not None else None
    sessions: list[SessionMeta] = []
    for path in session_dir.glob("*.jsonl"):
        try:
            meta = _read_meta_line(path)
        except OSError:
            continue
        if wanted is not None and meta.workdir != wanted:
            continue
        sessions.append(meta)
    sessions.sort(key=lambda m: m.updated_at, reverse=True)
    return sessions


def latest_session(
    session_dir: Path, workdir: Path | None = None
) -> Path | None:
    """Return the path of the most recently updated session, if any."""
    sessions = list_sessions(session_dir, workdir=workdir)
    if not sessions:
        return None
    return sessions[0].path


def find_session(session_dir: Path, id_prefix: str) -> Path:
    """Find a session by exact id or unique id prefix."""
    if not session_dir.is_dir():
        raise SessionNotFoundError(f"No sessions in {session_dir}")
    matches = sorted(p.stem for p in session_dir.glob(f"{id_prefix}*.jsonl"))
    if not matches:
        raise SessionNotFoundError(f"No session matching '{id_prefix}'")
    if len(matches) > 1:
        raise AmbiguousSessionError(id_prefix, matches)
    return session_dir / f"{matches[0]}.jsonl"


def export_markdown(
    meta: SessionMeta, messages: list[Message], path: Path
) -> None:
    """Export the conversation (user/assistant text) as a Markdown file."""
    lines = [
        f"# {meta.title or meta.id}",
        "",
        f"- workdir: `{meta.workdir}`",
        f"- model: `{meta.model}`",
        f"- updated: {meta.updated_at}",
        "",
    ]
    for msg in messages:
        if msg.role == "user" and msg.content:
            lines += ["## User", "", msg.content, ""]
        elif msg.role == "assistant" and msg.content:
            lines += ["## Assistant", "", msg.content, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_jsonl(
    meta: SessionMeta,
    messages: list[Message],
    path: Path,
    trace_path: Path | None = None,
) -> None:
    """Export the full-fidelity session log as a single JSONL file.

    Layout:

    1. a ``meta`` record (session metadata + export timestamp),
    2. every trace record in chronological order (LLM request bodies,
       usage, tool calls/results, errors) when the trace
       file exists — otherwise the raw conversation messages,
    3. a final ``messages_snapshot`` record with the exact message history
       as persisted.
    """
    meta_line = json.dumps(
        {
            "type": META_TYPE,
            **meta.model_dump(exclude={"path"}),
            "exported_at": _utc_now_iso(),
        },
        ensure_ascii=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(meta_line + "\n")
        trace_records = read_trace(trace_path) if trace_path is not None else []
        if trace_records:
            for record in trace_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            # Sessions from before tracing existed still export their messages.
            for msg in messages:
                f.write(msg.model_dump_json() + "\n")
        snapshot = json.dumps(
            {
                "type": SNAPSHOT_TYPE,
                "count": len(messages),
                "messages": [
                    json.loads(msg.model_dump_json()) for msg in messages
                ],
            },
            ensure_ascii=False,
        )
        f.write(snapshot + "\n")
