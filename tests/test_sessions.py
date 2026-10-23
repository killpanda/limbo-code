"""Tests for session storage (limbo.sessions)."""

from __future__ import annotations

import json
from pathlib import Path

from limbo.models import Message
from limbo.sessions import (
    SessionMeta,
    append_session_messages,
    load_session,
    save_session,
)


def make_meta(**overrides) -> SessionMeta:
    defaults = {
        "id": "20260720-103000-000000-1",
        "workdir": "/tmp/project",
        "model": "deepseek-v4-pro",
        "title": "hello world",
        "created_at": "2026-07-20T10:30:00+00:00",
        "updated_at": "2026-07-20T10:30:00+00:00",
    }
    defaults.update(overrides)
    return SessionMeta(**defaults)


def make_messages() -> list[Message]:
    return [
        Message(role="system", content="sys"),
        Message(role="user", content="hello world"),
        Message(role="assistant", content="hi there"),
    ]


def test_save_then_load_roundtrip(tmp_path: Path):
    path = tmp_path / f"{make_meta().id}.jsonl"
    meta = make_meta()
    messages = make_messages()

    save_session(path, meta, messages)
    loaded_meta, loaded_messages, _ = load_session(path)

    assert loaded_meta.id == meta.id
    assert loaded_meta.workdir == meta.workdir
    assert loaded_meta.title == meta.title
    assert [m.model_dump() for m in loaded_messages] == [
        m.model_dump() for m in messages
    ]


def test_append_session_messages_is_incremental(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    meta = make_meta()
    first = make_messages()
    save_session(path, meta, first)

    appended = [Message(role="assistant", content="more")]
    append_session_messages(path, appended)

    loaded_meta, loaded_messages, _ = load_session(path)
    assert loaded_meta.id == meta.id  # meta line untouched
    assert [m.content for m in loaded_messages] == [
        *[m.content for m in first], "more"
    ]
    # File only grew by the appended lines; the meta line still leads.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "meta"
    assert len(lines) == len(first) + len(appended) + 1


# -- compaction records ---------------------------------------------------------


def _compacted_messages() -> tuple[list[Message], str]:
    """[system, summary, *kept] plus the boundary (first kept) entry id."""
    from limbo.compaction import make_summary_message

    messages = [
        Message(role="system", content="sys"),
        make_summary_message("OLD SUMMARY"),
        Message(role="user", content="recent question"),
        Message(role="assistant", content="recent answer"),
    ]
    return messages, messages[2].id


def make_compaction_record(first_kept_entry_id: str, summary: str = "OLD SUMMARY"):
    from limbo.sessions import CompactionRecord

    return CompactionRecord(
        id="cmp-1",
        created_at="2026-07-20T11:00:00+00:00",
        first_kept_entry_id=first_kept_entry_id,
        summary=summary,
        tokens_before=120_000,
        tokens_after_estimate=22_000,
        trigger="auto",
    )


def test_load_drops_messages_before_boundary(tmp_path: Path):
    """A file that still carries pre-boundary messages gets them dropped."""
    from limbo.compaction import is_summary_message

    messages, boundary_id = _compacted_messages()
    stale = [
        messages[0],
        Message(role="user", content="ancient history"),
        Message(role="assistant", content="ancient reply"),
        *messages[1:],
    ]
    record = make_compaction_record(boundary_id)
    path = tmp_path / "s.jsonl"

    save_session(path, make_meta(), stale, compactions=[record])
    _, loaded, _ = load_session(path)

    contents = [m.content for m in loaded]
    assert "ancient history" not in contents
    assert loaded[0].role == "system"
    assert is_summary_message(loaded[1])
    assert loaded[1].content and "OLD SUMMARY" in loaded[1].content
    assert [m.id for m in loaded[2:]] == [m.id for m in messages[2:]]


# -- durable-session records (tool_intent / meta_patch) -----------------------




def test_full_rewrite_keeps_the_session_file_private(tmp_path: Path):
    """A later rewrite must not widen 0600 to the umask default (0644)."""
    import stat

    path = tmp_path / "s.jsonl"
    save_session(path, make_meta(), make_messages())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    save_session(path, make_meta(title="changed"), make_messages())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
