"""Tests for the append-only trace log (limbo.trace)."""

from __future__ import annotations

import json
import threading
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
    first = TraceLogger(path)
    first.log("session_start", session_id="s")
    first.close()  # drain the writer before the next instance appends
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


def test_flush_makes_records_readable_before_close(tmp_path: Path):
    logger = TraceLogger(tmp_path / "s.trace.jsonl")
    logger.log("user_message", turn=1, content="你好")
    logger.flush()  # barrier: everything logged so far is on disk
    records = read_trace(logger.path)
    assert [r["content"] for r in records] == ["你好"]
    logger.close()


def test_writer_preserves_fifo_order_under_bulk(tmp_path: Path):
    # log() only enqueues; the writer serializes/writes in the same order.
    logger = TraceLogger(tmp_path / "s.trace.jsonl")
    for i in range(300):
        logger.log("bulk", i=i, body="x" * 20_000)
    logger.close()
    records = read_trace(tmp_path / "s.trace.jsonl")
    assert [r["i"] for r in records] == list(range(300))


def test_flush_waits_for_writes_under_concurrency(tmp_path: Path):
    # Regression: task_done() used to run BEFORE the file write, so flush()
    # (queue.join) returned while records were still in flight — the
    # barrier let turn-end saves and /export see an incomplete trace.
    logger = TraceLogger(tmp_path / "s.trace.jsonl")

    def producer(n: int) -> None:
        for i in range(n):
            logger.log("bulk", i=i, body="x" * 50_000)

    threads = [threading.Thread(target=producer, args=(20,)) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    logger.flush()  # must block until every record is on disk
    records = read_trace(logger.path)
    assert len(records) == 80

    logger.log("tail", n=1)
    logger.flush()
    records = read_trace(logger.path)
    assert len(records) == 81
    logger.close()


def test_close_is_idempotent(tmp_path: Path):
    # close() is called from Agent.close() at every replacement point plus
    # teardown, so it must be safe to call repeatedly.
    logger = TraceLogger(tmp_path / "s.trace.jsonl")
    logger.log("session_start", session_id="s")
    logger.close()
    logger.close()  # no raise
    logger.log("late", x=1)  # dropped silently, fd is gone
    records = read_trace(tmp_path / "s.trace.jsonl")
    assert [r["type"] for r in records] == ["session_start"]

