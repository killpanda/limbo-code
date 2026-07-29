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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from limbo.models import Message

META_TYPE = "meta"


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
    # Populated by list/load; not written to disk.
    path: Path | None = None


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


def save_session(path: Path, meta: SessionMeta, messages: list[Message]) -> None:
    """Atomically write the whole session (meta line + message lines)."""
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
        for msg in messages:
            f.write(msg.model_dump_json() + "\n")
    os.replace(tmp_file, path)
    if not file_existed:
        path.chmod(0o600)


def _parse_meta(raw: dict[str, Any], fallback_id: str, path: Path) -> SessionMeta:
    data = {k: v for k, v in raw.items() if k != "type"}
    data.setdefault("id", fallback_id)
    return SessionMeta(path=path, **data)


def load_session(path: Path) -> tuple[SessionMeta, list[Message]]:
    """Load a session file, tolerating legacy files and malformed lines."""
    meta: SessionMeta | None = None
    messages: list[Message] = []
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
            try:
                messages.append(Message(**record))
            except ValueError:
                continue
    if meta is None:
        # Legacy file without a meta line.
        meta = SessionMeta(id=path.stem, path=path)
    return meta, messages


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
