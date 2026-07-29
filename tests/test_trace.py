"""Tests for the append-only trace log (limbo.trace)."""

from __future__ import annotations

import json
from pathlib import Path

from limbo.trace import TraceLogger, read_trace, trace_path_for


def test_trace_path_for_session_file():
    session = Path("/tmp/sessions/20260720-103000-1.jsonl")
    assert trace_path_for(session) == Path(
        "/tmp/sessions/traces/20260720-103000-1.trace.jsonl"
    )


def test_log_writes_jsonl_records(tmp_path: Path):
    logger = TraceLogger(tmp_path / "s.trace.jsonl")
    logger.log("session_start", session_id="s", config={"model": "m"})
    logger.log("user_message", turn=1, content="你好")
    logger.close()

    lines = (tmp_path / "s.trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["type"] == "session_start"
    assert first["session_id"] == "s"
    assert first["config"] == {"model": "m"}
    assert "ts" in first
    second = json.loads(lines[1])
    assert second["content"] == "你好"  # ensure_ascii=False


def test_log_appends_across_instances(tmp_path: Path):
    path = tmp_path / "s.trace.jsonl"
    TraceLogger(path).log("session_start", session_id="s")
    # A resumed session reopens the same trace file and appends.
    logger = TraceLogger(path)
    logger.log("session_start", session_id="s", resumed=True)
    logger.close()

    records = read_trace(path)
    assert len(records) == 2
    assert records[1]["resumed"] is True


def test_log_serializes_arbitrary_values(tmp_path: Path):
    path = tmp_path / "s.trace.jsonl"
    logger = TraceLogger(path)
    logger.log("weird", value=object())  # falls back to str via default=
    logger.close()

    record = read_trace(path)[0]
    assert record["type"] == "weird"
    assert isinstance(record["value"], str)


def test_read_trace_skips_malformed_lines(tmp_path: Path):
    path = tmp_path / "s.trace.jsonl"
    path.write_text(
        '{"type": "ok"}\nnot json\n\n["a", "list"]\n{"type": "ok2"}\n',
        encoding="utf-8",
    )
    records = read_trace(path)
    assert [r["type"] for r in records] == ["ok", "ok2"]


def test_read_trace_missing_file(tmp_path: Path):
    assert read_trace(tmp_path / "missing.jsonl") == []
