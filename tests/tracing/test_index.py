"""Tests for the incremental trace index (limbo.tracing.index)."""

from __future__ import annotations

import json
from pathlib import Path

from limbo.tracing.index import TraceIndex


def write_lines(path: Path, records: list[dict], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_refresh_is_incremental_and_reads_full_records(tmp_path):
    path = tmp_path / "s.trace.jsonl"
    write_lines(
        path, [{"type": "session_start"}, {"type": "llm_request", "body": {"messages": []}}]
    )
    index = TraceIndex(path)
    assert index.refresh() == 2
    assert index.refresh() == 0

    write_lines(path, [{"type": "llm_response", "content": "hi"}])
    assert index.refresh() == 1
    records = index.records
    assert [r.summary["type"] for r in records] == ["session_start", "llm_request", "llm_response"]

    full = index.read(records[2].offset)
    assert full == {"type": "llm_response", "content": "hi"}
    # Only indexed offsets are served (no arbitrary seeks).
    assert index.read(records[2].offset + 1) is None


