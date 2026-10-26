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




def test_request_history_is_encoded_per_message_with_identical_output(tmp_path: Path):
    """The writer encodes a body's history message by message (so it never
    holds the GIL for a whole multi-MB body); the line must not change."""
    import json

    from limbo.trace import _encode

    record = {
        "ts": "t",
        "type": "llm_request",
        "body": {
            "model": "m",
            "messages": [
                {"role": "user", "content": 'say "hi" 中文\n'},
                {"role": "tool", "content": "out", "extra": Path("p")},
            ],
        },
    }
    assert _encode(record) == json.dumps(record, ensure_ascii=False, default=str)
    responses = {"type": "llm_request", "body": {"input": [{"a": 1}, {"b": [1, 2]}]}}
    assert _encode(responses) == json.dumps(responses, ensure_ascii=False)

    logger = TraceLogger(tmp_path / "s.trace.jsonl")
    logger.log("llm_request", body=record["body"])
    logger.close()
    (written,) = read_trace(tmp_path / "s.trace.jsonl")
    assert written["body"]["messages"][0]["content"] == 'say "hi" 中文\n'
