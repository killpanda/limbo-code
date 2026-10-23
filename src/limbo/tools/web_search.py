"""Web search tool (backend chosen by [web_search] provider)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError, truncate_output
from limbo.websearch import SearchError, SearchProvider, SearchResult

MAX_QUERY_CHARS = 128


class WebSearchTool(BaseTool):
    name = "web_search"
    # Read-only: replaying it after a crash (or calling it under /plan)
    # changes nothing.
    replay = "safe"

    def __init__(self, workdir: Path, provider: SearchProvider, default_count: int = 10):
        super().__init__(workdir)
        self.provider = provider
        self.default_count = max(1, min(default_count, provider.max_count))
        self.description = (
            "Search the web for up-to-date information (docs, releases, news, "
            "error messages). Returns ranked results with title, URL, date "
            f"and a summary. Queries are at most {MAX_QUERY_CHARS} characters."
            + provider.query_help
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "count": {
                    "type": "integer",
                    "description": (
                        f"Number of results (1-{provider.max_count}, "
                        f"default {self.default_count})."
                    ),
                },
            },
            "required": ["query"],
        }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        query = query.strip()
        if len(query) > MAX_QUERY_CHARS:
            raise ToolError(
                f"query is {len(query)} characters; the limit is {MAX_QUERY_CHARS}"
            )
        count = arguments.get("count", self.default_count)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ToolError("count must be an integer")
        count = max(1, min(count, self.provider.max_count))
        try:
            results = self.provider.search(query, count)
        except SearchError as e:
            raise ToolError(str(e)) from e
        if not results:
            return ToolResult(success=True, output=f"No results for: {query}")
        return ToolResult(success=True, output=truncate_output(format_results(results)))


def format_results(results: list[SearchResult]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        lines = [f"[{i}] {r.title or r.url}", r.url]
        meta = " · ".join(p for p in (r.site_name, r.date) if p)
        if meta:
            lines.append(meta)
        if r.summary:
            lines.append(r.summary)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
