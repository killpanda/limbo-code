"""Tests for the incremental trace index (limbo.tracing.index)."""

from __future__ import annotations

import json
from pathlib import Path

from limbo.tracing.index import PREVIEW_CHARS, TraceIndex, summarize


def write_lines(path: Path, records: list[dict], mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_summarize_drops_bulky_fields_and_keeps_previews():
    body = {"model": "m", "messages": [{"role": "system"}, {"role": "user"}]}
    summary = summarize({"type": "llm_request", "turn": 1, "body": body})
    assert "body" not in summary
    assert summary["model"] == "m"
    assert summary["message_count"] == 2

    long_output = "x" * (PREVIEW_CHARS + 50)
    summary = summarize({"type": "tool_result", "output": long_output, "duration": 0.1})
    assert "output" not in summary
    assert summary["output_preview"].endswith("…")
    assert len(summary["output_preview"]) == PREVIEW_CHARS + 1
    assert summary["duration"] == 0.1


def test_summarize_strips_tool_call_arguments():
    summary = summarize(
        {
            "type": "llm_response",
            "tool_calls": [{"id": "c1", "name": "read", "arguments": {"p": 1}}],
        }
    )
    assert summary["tool_calls"] == [{"id": "c1", "name": "read"}]


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


def test_partial_trailing_line_waits_for_next_refresh(tmp_path):
    path = tmp_path / "s.trace.jsonl"
    write_lines(path, [{"type": "a"}])
    with path.open("a", encoding="utf-8") as f:
        f.write('{"type": "b"')  # writer thread mid-write
    index = TraceIndex(path)
    assert index.refresh() == 1
    with path.open("a", encoding="utf-8") as f:
        f.write("}\n")
    assert index.refresh() == 1
    assert [r.summary["type"] for r in index.records] == ["a", "b"]


def test_malformed_lines_are_skipped(tmp_path):
    path = tmp_path / "s.trace.jsonl"
    path.write_text('{"type": "a"}\nnot json\n[1, 2]\n\n{"type": "b"}\n', encoding="utf-8")
    index = TraceIndex(path)
    assert index.refresh() == 2


def test_truncated_file_is_reindexed(tmp_path):
    path = tmp_path / "s.trace.jsonl"
    write_lines(path, [{"type": "a"}, {"type": "b"}])
    index = TraceIndex(path)
    index.refresh()
    write_lines(path, [{"type": "c"}], mode="w")
    index.refresh()
    assert [r.summary["type"] for r in index.records] == ["c"]


def test_missing_file_is_empty(tmp_path):
    index = TraceIndex(tmp_path / "nope.trace.jsonl")
    assert index.refresh() == 0
    assert index.records == []


def test_previous_request_skips_compaction(tmp_path):
    path = tmp_path / "s.trace.jsonl"
    write_lines(
        path,
        [
            {"type": "llm_request", "body": {}},
            {"type": "llm_request", "kind": "compaction", "body": {}},
            {"type": "llm_request", "body": {}},
        ],
    )
    index = TraceIndex(path)
    index.refresh()
    first, _, third = index.records
    assert index.previous_request(third.offset) == first.offset
    assert index.previous_request(first.offset) is None
