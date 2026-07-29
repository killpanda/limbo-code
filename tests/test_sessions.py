"""Tests for session storage (limbo.sessions)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from limbo.models import Message
from limbo.sessions import (
    AmbiguousSessionError,
    SessionMeta,
    SessionNotFoundError,
    export_jsonl,
    export_markdown,
    find_session,
    latest_session,
    list_sessions,
    load_session,
    save_session,
)
from limbo.trace import TraceLogger


def make_meta(**overrides) -> SessionMeta:
    defaults = {
        "id": "20260720-103000-000000-1",
        "workdir": "/tmp/project",
        "model": "deepseek-chat",
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
    loaded_meta, loaded_messages = load_session(path)

    assert loaded_meta.id == meta.id
    assert loaded_meta.workdir == meta.workdir
    assert loaded_meta.title == meta.title
    assert [m.model_dump() for m in loaded_messages] == [
        m.model_dump() for m in messages
    ]


def test_save_writes_meta_as_first_line(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    save_session(path, make_meta(), make_messages())

    first_line = json.loads(path.read_text().splitlines()[0])
    assert first_line["type"] == "meta"
    assert first_line["title"] == "hello world"


def test_save_is_atomic_and_sets_permissions(tmp_path: Path):
    path = tmp_path / "x.jsonl"
    save_session(path, make_meta(), make_messages())

    assert not (tmp_path / "x.tmp").exists()
    assert (path.stat().st_mode & 0o777) == 0o600


def test_load_legacy_file_without_meta(tmp_path: Path):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"role":"system","content":"sys"}\n'
        '{"role":"user","content":"hi"}\n'
    )

    meta, messages = load_session(path)

    assert meta.id == "legacy"  # falls back to file stem
    assert meta.title == ""
    assert [m.role for m in messages] == ["system", "user"]


def test_load_skips_blank_and_malformed_lines(tmp_path: Path):
    path = tmp_path / "messy.jsonl"
    path.write_text(
        "\n"
        "not json\n"
        '{"role":"user","content":"ok"}\n'
    )

    _, messages = load_session(path)
    assert [m.content for m in messages] == ["ok"]


def _write_session_file(path: Path, meta: SessionMeta) -> None:
    """Write a meta-only session file, preserving the given timestamps."""
    meta_line = json.dumps(
        {"type": "meta", **meta.model_dump(exclude={"path"})},
        ensure_ascii=False,
    )
    path.write_text(meta_line + "\n")


def test_list_sessions_sorted_by_updated_at_desc(tmp_path: Path):
    for i, updated in enumerate(["2026-07-18", "2026-07-20", "2026-07-19"]):
        meta = make_meta(id=f"s{i}", updated_at=f"{updated}T00:00:00+00:00")
        _write_session_file(tmp_path / f"s{i}.jsonl", meta)

    sessions = list_sessions(tmp_path)
    assert [s.id for s in sessions] == ["s1", "s2", "s0"]
    assert all(s.path is not None for s in sessions)


def test_list_sessions_filters_by_workdir(tmp_path: Path):
    save_session(tmp_path / "a.jsonl", make_meta(id="a", workdir="/proj/a"), [])
    save_session(tmp_path / "b.jsonl", make_meta(id="b", workdir="/proj/b"), [])

    sessions = list_sessions(tmp_path, workdir=Path("/proj/a"))
    assert [s.id for s in sessions] == ["a"]


def test_list_sessions_includes_legacy_files(tmp_path: Path):
    (tmp_path / "old.jsonl").write_text('{"role":"user","content":"hi"}\n')

    sessions = list_sessions(tmp_path)
    assert len(sessions) == 1
    assert sessions[0].id == "old"


def test_list_sessions_ignores_tmp_files(tmp_path: Path):
    save_session(tmp_path / "a.jsonl", make_meta(id="a"), [])
    (tmp_path / "a.tmp").write_text("partial")

    assert [s.id for s in list_sessions(tmp_path)] == ["a"]


def test_latest_session_returns_most_recent(tmp_path: Path):
    _write_session_file(
        tmp_path / "a.jsonl",
        make_meta(id="a", updated_at="2026-07-18T00:00:00+00:00"),
    )
    _write_session_file(
        tmp_path / "b.jsonl",
        make_meta(id="b", updated_at="2026-07-20T00:00:00+00:00"),
    )

    assert latest_session(tmp_path) == tmp_path / "b.jsonl"
    assert latest_session(tmp_path / "empty") is None


def test_latest_session_respects_workdir_filter(tmp_path: Path):
    _write_session_file(
        tmp_path / "a.jsonl",
        make_meta(id="a", workdir="/proj/a", updated_at="2026-07-20T00:00:00+00:00"),
    )
    _write_session_file(
        tmp_path / "b.jsonl",
        make_meta(id="b", workdir="/proj/b", updated_at="2026-07-19T00:00:00+00:00"),
    )

    assert latest_session(tmp_path, workdir=Path("/proj/b")) == tmp_path / "b.jsonl"


def test_find_session_by_id_prefix(tmp_path: Path):
    save_session(tmp_path / "abc123.jsonl", make_meta(id="abc123"), [])

    assert find_session(tmp_path, "abc") == tmp_path / "abc123.jsonl"


def test_find_session_not_found(tmp_path: Path):
    with pytest.raises(SessionNotFoundError):
        find_session(tmp_path, "nope")


def test_find_session_ambiguous_prefix(tmp_path: Path):
    save_session(tmp_path / "abc1.jsonl", make_meta(id="abc1"), [])
    save_session(tmp_path / "abc2.jsonl", make_meta(id="abc2"), [])

    with pytest.raises(AmbiguousSessionError) as exc_info:
        find_session(tmp_path, "abc")
    assert "abc1" in str(exc_info.value)
    assert "abc2" in str(exc_info.value)


def test_export_markdown(tmp_path: Path):
    meta = make_meta(title="Fix the bug")
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="please fix"),
        Message(role="assistant", content="done"),
        Message(role="tool", content="tool output", tool_call_id="c1"),
    ]
    out = tmp_path / "export.md"

    export_markdown(meta, messages, out)

    text = out.read_text()
    assert "# Fix the bug" in text
    assert "## User" in text and "please fix" in text
    assert "## Assistant" in text and "done" in text
    # system prompt and raw tool outputs are not exported
    assert "sys" not in text
    assert "tool output" not in text


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_export_jsonl_merges_trace_and_snapshot(tmp_path: Path):
    meta = make_meta(title="Fix the bug")
    messages = make_messages()
    trace_path = tmp_path / "s.trace.jsonl"
    logger = TraceLogger(trace_path)
    logger.log("session_start", session_id=meta.id)
    logger.log("llm_request", turn=1, iteration=1, body={"model": "m"})
    logger.log("tool_result", turn=1, name="read", success=True)
    logger.close()
    out = tmp_path / "export.jsonl"

    export_jsonl(meta, messages, out, trace_path=trace_path)

    records = _read_jsonl(out)
    assert records[0]["type"] == "meta"
    assert records[0]["id"] == meta.id
    assert "exported_at" in records[0]
    # Trace records follow in chronological order.
    assert [r["type"] for r in records[1:4]] == [
        "session_start",
        "llm_request",
        "tool_result",
    ]
    assert records[2]["body"] == {"model": "m"}
    # Final record is the message snapshot.
    snapshot = records[-1]
    assert snapshot["type"] == "messages_snapshot"
    assert snapshot["count"] == len(messages)
    assert snapshot["messages"][1]["content"] == "hello world"


def test_export_jsonl_without_trace_falls_back_to_messages(tmp_path: Path):
    meta = make_meta()
    messages = make_messages()
    out = tmp_path / "export.jsonl"

    export_jsonl(meta, messages, out, trace_path=tmp_path / "missing.jsonl")

    records = _read_jsonl(out)
    assert records[0]["type"] == "meta"
    # Raw messages stand in for the missing trace.
    assert records[1]["role"] == "system"
    assert records[-1]["type"] == "messages_snapshot"


def test_export_jsonl_every_line_is_valid_json(tmp_path: Path):
    meta = make_meta()
    out = tmp_path / "export.jsonl"
    export_jsonl(meta, make_messages(), out, trace_path=None)

    # _read_jsonl raises on any malformed line.
    assert len(_read_jsonl(out)) >= 2
