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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from limbo.compaction import is_summary_message, make_summary_message
from limbo.goal import GoalState
from limbo.models import Attachment, Message
from limbo.trace import read_trace

META_TYPE = "meta"
SNAPSHOT_TYPE = "messages_snapshot"
COMPACTION_TYPE = "compaction"
# Durable-session records (pi-durable style "commit intent, then effect"):
# appended between message lines, never rewritten in place.
TOOL_INTENT_TYPE = "tool_intent"
META_PATCH_TYPE = "meta_patch"
# Queued user messages (steer / follow-up inbox): "queued" when offered,
# "dequeued" once injected into history or cancelled.
QUEUE_TYPE = "queue"
# Subagent task board events (limbo.subagents.board): folded into the
# board by the supervisor; a "snapshot" op restarts the fold after a
# full rewrite dropped the event history.
TASK_TYPE = "task"


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


class UsageEntry(BaseModel):
    """Accumulated usage for one ``provider/model`` in a session."""

    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


class SessionMeta(BaseModel):
    """Metadata describing a session (stored as the first JSONL line)."""

    id: str
    workdir: str = ""
    model: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    # Active/finished closed-loop goal (LIM-40); restored on resume (D4/D6).
    goal: GoalState | None = None
    # Code Mode presentation (deepseek-harness parity): whether the
    # model-facing catalog was collapsed to `run_code`; restored on resume.
    code_mode: bool = False
    # Usage ledger keyed by "provider/model": every billed call (turns,
    # compaction summaries, /btw) adds to it; persisted via meta_patch so
    # the token counter survives resume.
    usage: dict[str, UsageEntry] = Field(default_factory=dict)
    # Plan mode (/plan): the read-only tool hook is mounted; restored on resume.
    plan_mode: bool = False
    # Whether tool calls are journaled as ``tool_intent`` records before
    # they run. Legacy sessions lack it, so a dangling call there cannot be
    # told apart as "never started" vs "interrupted mid-run".
    tool_intents: bool = False
    # Lineage (/fork, /handoff): the session this one branched from, and the
    # id of the last parent message it copied (None for a /handoff, which
    # starts a fresh context from a note).
    parent_session: str | None = None
    fork_at: str | None = None
    # Subagent sessions: "subagent" plus the task it executes. They live in
    # ``<parent id>.agents/`` beside the parent, so listing never sees them.
    role: str = "main"
    task_id: str | None = None
    # Populated by list/load; not written to disk.
    path: Path | None = None


class CompactionRecord(BaseModel):
    """One compaction event, stored as a JSONL line in the session file.

    ``first_kept_entry_id`` marks the keep boundary (pi's entry model):
    on load, messages before that entry are dropped and the stored
    ``summary`` is re-inserted as the conversation summary. Compaction is
    non-destructive: the record is appended and the summarized messages
    stay in the file, so the boundary selects the active range on load.
    """

    id: str
    created_at: str = ""
    first_kept_entry_id: str = ""
    summary: str = ""
    tokens_before: int | None = None
    tokens_after_estimate: int | None = None
    trigger: str = "auto"  # "auto" | "manual" | "overflow"


class ToolIntentRecord(BaseModel):
    """A tool call about to execute, appended before its side effects run.

    Recovery reads it to tell "never started" (no intent) from "may have
    partially run" (intent without a result). ``replay`` is the tool's
    declared policy: ``safe`` calls are re-run on resume, ``unsafe`` ones
    get an interrupted result instead.
    """

    call_id: str
    name: str
    replay: str = "unsafe"
    created_at: str = ""


class QueuedMessageRecord(BaseModel):
    """A user message queued mid-turn, journaled so a crash keeps it."""

    id: str
    text: str
    attachments: list[Attachment] = []
    mode: str = "steer"  # "steer" | "follow_up"
    # "user" or "subagent" (a notice delivered to the main agent); refs
    # are the notice ids it carries, so a crash between queueing a notice
    # and marking it reported never delivers it twice.
    source: str = "user"
    refs: list[str] = []


@dataclass
class LoadedSession:
    """Everything :func:`load_session_full` reconstructs from one file."""

    meta: SessionMeta
    messages: list[Message]
    compactions: list[CompactionRecord] = field(default_factory=list)
    intents: list[ToolIntentRecord] = field(default_factory=list)
    # Messages still queued when the file was last written, in FIFO order.
    queue: list[QueuedMessageRecord] = field(default_factory=list)
    # Raw task board records, in file order (folded by the supervisor).
    tasks: list[dict[str, Any]] = field(default_factory=list)


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
    """Atomically (re)write the whole session: meta + compactions + messages.

    The full rewrite is the heavyweight path — used on session creation and
    whenever the on-disk history changes shape (compaction) or the meta
    line must be refreshed (title/code_mode). Steady-state persistence goes
    through :func:`append_session_messages`, which only appends new
    messages, so long sessions do not rewrite megabytes per turn.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    meta.updated_at = _utc_now_iso()
    meta_line = json.dumps(
        {"type": META_TYPE, **meta.model_dump(exclude={"path"})},
        ensure_ascii=False,
    )
    tmp_file = path.with_suffix(".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        # The temp file is created with the umask, so on the default 0o022
        # it would be 0o644. Tighten it before the replace: the previous
        # chmod only ran on first creation, and every later full rewrite
        # (compaction, resume repair) silently widened the session — which
        # holds prompts, code and tool output — to world-readable.
        os.fchmod(f.fileno(), 0o600)
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


def append_session_messages(path: Path, messages: list[Message]) -> None:
    """Append new messages to an existing session file (incremental save).

    The cheap steady-state write: the meta line and all previously saved
    messages stay untouched, only ``messages`` are appended. Requires the
    file to already exist with a meta first line (created by
    :func:`save_session`) — a missing file raises instead of silently
    creating a meta-less session, so the full-rewrite-then-append contract
    stays explicit. Permissions are preserved (no re-chmod). A crash
    during the append can leave a torn last line (skipped on load) and
    loses at most the in-flight increment — narrower than the atomic
    rewrite, which is why callers fall back to a full rewrite after a
    failed append.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot append to missing session file: {path}"
        )
    with path.open("a", encoding="utf-8") as f:
        for msg in messages:
            f.write(msg.model_dump_json() + "\n")


def append_session_records(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Append typed (``{"type": ...}``) records to an existing session file.

    Same contract as :func:`append_session_messages`: the file must already
    exist, and a crash mid-append leaves at most a torn last line, which
    loading skips.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot append to missing session file: {path}"
        )
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tool_intent_record(call_id: str, name: str, replay: str) -> dict[str, Any]:
    """Serialize a :class:`ToolIntentRecord` as an appendable line."""
    intent = ToolIntentRecord(
        call_id=call_id, name=name, replay=replay, created_at=_utc_now_iso()
    )
    return {"type": TOOL_INTENT_TYPE, **intent.model_dump()}


def meta_patch_record(meta: SessionMeta) -> dict[str, Any]:
    """Serialize the current meta as a patch line (later patches win)."""
    return {
        "type": META_PATCH_TYPE,
        **meta.model_dump(exclude={"path", "id"}),
    }


def queued_record(item: QueuedMessageRecord) -> dict[str, Any]:
    """Serialize a newly queued message."""
    return {"type": QUEUE_TYPE, "op": "queued", **item.model_dump()}


def dequeued_record(ids: Sequence[str]) -> dict[str, Any]:
    """Serialize queue items leaving the queue (injected or cancelled)."""
    return {"type": QUEUE_TYPE, "op": "dequeued", "ids": list(ids)}


def compaction_record(record: CompactionRecord) -> dict[str, Any]:
    """Serialize a compaction as an appendable session line."""
    return {"type": COMPACTION_TYPE, **record.model_dump()}


def rewrite_meta_line(path: Path, meta: SessionMeta) -> None:
    """Atomically replace the meta line, keeping every other record.

    Listing reads only the first line, so a title change must land there;
    everything else in the journal (messages, compactions, intents) is
    copied verbatim. ``meta_patch`` lines are dropped: the new meta line
    already carries their merged state, and an older patch would otherwise
    override it on load.
    """
    meta.updated_at = _utc_now_iso()
    meta_line = json.dumps(
        {"type": META_TYPE, **meta.model_dump(exclude={"path"})},
        ensure_ascii=False,
    )
    tmp_file = path.with_suffix(".tmp")
    with path.open("r", encoding="utf-8") as src, tmp_file.open(
        "w", encoding="utf-8"
    ) as dst:
        os.fchmod(dst.fileno(), 0o600)
        dst.write(meta_line + "\n")
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                record = None
            if isinstance(record, dict) and record.get("type") in (
                META_TYPE,
                META_PATCH_TYPE,
            ):
                continue
            dst.write(stripped + "\n")
    os.replace(tmp_file, path)


def _parse_meta(raw: dict[str, Any], fallback_id: str, path: Path) -> SessionMeta:
    data = {k: v for k, v in raw.items() if k != "type"}
    data.setdefault("id", fallback_id)
    return SessionMeta(path=path, **data)


def _apply_compactions(
    messages: list[Message], compactions: list[CompactionRecord]
) -> list[Message]:
    """Reconstruct the post-compaction history from the latest record.

    Only the most recent record participates in reconstruction (older ones
    stay on disk for audit). Compaction is non-destructive: the summarized
    messages stay in the journal and this function selects the active range
    — everything from the record's ``first_kept_entry_id`` on, behind the
    summary. If the boundary id is missing (hand-edited file, partial
    write), the record is ignored with a warning — better uncompacted than
    data loss.
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
    # A full rewrite stores the summary message before the boundary; the
    # append path stores only the record, and a stored summary may belong to
    # an older compaction. Reuse the stored message (keeping its id) only
    # when it matches the latest record; otherwise rebuild it.
    dropped = messages[len(head) : boundary]
    stored = next((m for m in reversed(dropped) if is_summary_message(m)), None)
    rebuilt = make_summary_message(latest.summary)
    if stored is not None and (
        not latest.summary or stored.content == rebuilt.content
    ):
        summary_msg = stored
    else:
        summary_msg = rebuilt
    return [*head, summary_msg, *kept]


def load_session(
    path: Path,
) -> tuple[SessionMeta, list[Message], list[CompactionRecord]]:
    """Load a session file, tolerating legacy files and malformed lines.

    Returns the meta, the reconstructed message history (with compaction
    applied), and the raw compaction records (the agent uses the latest one
    to restore its iterative-summary chain). See :func:`load_session_full`
    for the durable-session records as well.
    """
    loaded = load_session_full(path)
    return loaded.meta, loaded.messages, loaded.compactions


def load_session_full(path: Path) -> LoadedSession:
    """Load every record of a session file.

    ``meta_patch`` lines are merged over the meta line in file order;
    ``tool_intent`` lines are returned for crash recovery; records of an
    unknown ``type`` (written by a newer version) are skipped.
    """
    meta: SessionMeta | None = None
    patches: list[dict[str, Any]] = []
    messages: list[Message] = []
    compactions: list[CompactionRecord] = []
    intents: list[ToolIntentRecord] = []
    queued: dict[str, QueuedMessageRecord] = {}
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Torn tail from a crashed incremental append: skip, but say
                # so — a silently dropped message is hard to diagnose.
                warnings.warn(
                    f"Skipping malformed line in session {path}: {line[:60]!r}",
                    stacklevel=2,
                )
                continue
            if not isinstance(record, dict):
                continue
            kind = record.get("type")
            data = {k: v for k, v in record.items() if k != "type"}
            if kind is None:
                try:
                    messages.append(Message(**record))
                except ValueError:
                    continue
            elif kind == META_TYPE:
                meta = _parse_meta(record, path.stem, path)
            elif kind == META_PATCH_TYPE:
                patches.append(data)
            elif kind == COMPACTION_TYPE:
                try:
                    compactions.append(CompactionRecord(**data))
                except ValueError:
                    continue
            elif kind == QUEUE_TYPE:
                if data.get("op") == "queued":
                    try:
                        item = QueuedMessageRecord(
                            **{k: v for k, v in data.items() if k != "op"}
                        )
                    except ValueError:
                        continue
                    queued[item.id] = item
                elif data.get("op") == "dequeued":
                    for item_id in data.get("ids") or []:
                        queued.pop(item_id, None)
            elif kind == TASK_TYPE:
                if data.get("op") == "snapshot":
                    tasks.clear()
                tasks.append(record)
            elif kind == TOOL_INTENT_TYPE:
                try:
                    intents.append(ToolIntentRecord(**data))
                except ValueError:
                    continue
            # Any other type: a record from a newer version — skip it.
    if meta is None:
        # Legacy file without a meta line.
        meta = SessionMeta(id=path.stem, path=path)
    for patch in patches:
        merged = {**meta.model_dump(exclude={"path"}), **patch, "id": meta.id}
        try:
            meta = SessionMeta(path=path, **merged)
        except ValueError:
            continue
    return LoadedSession(
        meta=meta,
        messages=_apply_compactions(messages, compactions),
        compactions=compactions,
        intents=intents,
        queue=list(queued.values()),
        tasks=tasks,
    )


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
    # Surface the file mtime as updated_at: incremental saves append to the
    # file without rewriting the meta line, so the meta record's own
    # updated_at goes stale (it still carries the creation / last-full-
    # rewrite time), while the file's mtime is exactly "when this session
    # was last written". Callers (picker, sorting) see fresh values without
    # knowing the incremental-save detail.
    for meta in sessions:
        meta_path = meta.path
        if meta_path is None:
            continue
        try:
            meta.updated_at = datetime.fromtimestamp(
                meta_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
        except OSError:
            pass  # keep the stale meta value; the file vanished mid-list
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
