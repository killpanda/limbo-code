"""Tests for the append-only trace log (limbo.trace)."""

from __future__ import annotations

from pathlib import Path

from limbo.trace import TraceLogger, read_trace


def test_writer_preserves_fifo_order_under_bulk(tmp_path: Path):
    # log() only enqueues; the writer serializes/writes in the same order.
    logger = TraceLogger(tmp_path / "s.trace.jsonl")
    for i in range(300):
        logger.log("bulk", i=i, body="x" * 20_000)
    logger.close()
    records = read_trace(tmp_path / "s.trace.jsonl")
    assert [r["i"] for r in records] == list(range(300))


