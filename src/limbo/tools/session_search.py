"""The ``session_search`` tool (daily mode): recall past conversations.

Episodic memory next to the curated ``memory`` files: it scans the session
JSONL files (no index, no LLM call) and returns real message snippets.
Subagent sessions live in ``<id>.agents/`` subdirectories and are never
searched.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError, truncate_output

MAX_FILES = 500
MAX_HITS = 10
SNIPPET_CHARS = 300
READ_CHARS = 2000
WINDOW = 4


@dataclass
class _Session:
    id: str
    title: str
    date: str
    daily: bool
    messages: list[dict[str, Any]] = field(default_factory=list)


def _load(path: Path) -> _Session | None:
    meta: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line
        if not isinstance(rec, dict):
            continue
        kind = rec.get("type")
        if kind == "meta":
            meta = rec
        elif kind == "meta_patch":
            meta.update({k: v for k, v in rec.items() if k != "type"})
        elif kind is None and rec.get("role") in ("user", "assistant") and rec.get("content"):
            messages.append(rec)
    if not meta:
        return None
    return _Session(
        id=str(meta.get("id") or path.stem),
        title=str(meta.get("title") or ""),
        date=str(meta.get("created_at") or "")[:10],
        daily=bool(meta.get("daily_mode")),
        messages=messages,
    )


def _snippet(text: str, terms: list[str]) -> str:
    low = text.lower()
    at = min((low.find(t) for t in terms if t in low), default=0)
    start = max(0, at - SNIPPET_CHARS // 3)
    piece = re.sub(r"\s+", " ", text[start : start + SNIPPET_CHARS]).strip()
    return ("…" if start else "") + piece + ("…" if start + SNIPPET_CHARS < len(text) else "")


class SessionSearchTool(BaseTool):
    name = "session_search"
    replay = "safe"
    description = (
        "Search and reread past conversations with the user. Use it when the user "
        "refers to earlier work (\"what did we decide about X\", \"continue the "
        "report from last week\") or a past session likely holds the answer. "
        "With query: sessions whose messages contain every word, newest first, "
        "with snippets and message indices. With session_id: read that session "
        "(around=<index> for the messages near one hit). Searches daily sessions "
        "unless include_coding is true."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words that must all appear."},
            "session_id": {"type": "string", "description": "A session id (or prefix) to read."},
            "around": {"type": "integer", "description": "With session_id: a message index."},
            "include_coding": {
                "type": "boolean",
                "description": "Also search coding sessions (default false).",
            },
        },
    }

    def __init__(self, workdir: Path, session_dir: Callable[[], Path]):
        super().__init__(workdir)
        self._session_dir = session_dir

    def _files(self) -> list[Path]:
        directory = self._session_dir()
        if not directory.is_dir():
            return []
        files = sorted(directory.glob("*.jsonl"), key=_mtime, reverse=True)
        return files[:MAX_FILES]

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        session_id = arguments.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return self._read(session_id.strip(), arguments.get("around"))
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("give query (words to find) or session_id (a session to read)")
        terms = [t.lower() for t in query.split()]
        include_coding = bool(arguments.get("include_coding"))
        blocks: list[str] = []
        for path in self._files():
            session = _load(path)
            if session is None or not (session.daily or include_coding):
                continue
            hits = [
                (i, m) for i, m in enumerate(session.messages)
                if all(t in str(m["content"]).lower() for t in terms)
            ]
            if not hits:
                continue
            lines = [
                f"session {session.id} · {session.date} · {session.title or '(untitled)'}"
                + ("" if session.daily else " · coding")
            ]
            for i, m in hits[:3]:
                lines.append(f"  [{i}] {m['role']}: {_snippet(str(m['content']), terms)}")
            if len(hits) > 3:
                lines.append(f"  … {len(hits) - 3} more matching messages")
            blocks.append("\n".join(lines))
            if len(blocks) >= MAX_HITS:
                break
        if not blocks:
            return ToolResult(success=True, output=f"No past sessions mention: {query}")
        return ToolResult(
            success=True,
            output=truncate_output(
                "\n\n".join(blocks)
                + "\n\nRead one with session_id (and around=<index> for context)."
            ),
        )

    def _read(self, ref: str, around: Any) -> ToolResult:
        matches = [p for p in self._files() if p.stem.startswith(ref)]
        if not matches:
            raise ToolError(f"no session matches {ref!r}")
        if len(matches) > 1 and not any(p.stem == ref for p in matches):
            raise ToolError(f"{len(matches)} sessions match {ref!r}; give more of the id")
        path = next((p for p in matches if p.stem == ref), matches[0])
        session = _load(path)
        if session is None:
            raise ToolError(f"cannot read session {ref!r}")
        messages = list(enumerate(session.messages))
        if isinstance(around, int) and not isinstance(around, bool):
            messages = messages[max(0, around - WINDOW) : around + WINDOW + 1]
        elif len(messages) > 2 * WINDOW + 2:
            head, tail = messages[: WINDOW + 1], messages[-(WINDOW + 1) :]
            skipped = len(messages) - len(head) - len(tail)
            messages = [*head, (-1, {"role": "…", "content": f"{skipped} messages skipped"}),
                        *tail]
        lines = [f"session {session.id} · {session.date} · {session.title or '(untitled)'}"]
        for i, m in messages:
            text = str(m["content"])
            if len(text) > READ_CHARS:
                text = text[:READ_CHARS] + "…"
            label = f"[{i}] " if i >= 0 else ""
            lines.append(f"{label}{m['role']}: {text}")
        return ToolResult(success=True, output=truncate_output("\n\n".join(lines)))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
