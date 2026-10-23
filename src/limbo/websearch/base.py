"""The web-search provider seam.

A provider turns a query into a ranked list of :class:`SearchResult`; the
``web_search`` tool (limbo.tools.web_search) formats them for the model.
Adding a provider is one subclass plus one entry in
``limbo.websearch.PROVIDERS`` — the tool and the config never change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SearchError(Exception):
    """A search failed; the message is shown to the model as the tool error."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    # Model-oriented text for the hit (a provider's longer summary when it
    # has one, else its display snippet).
    summary: str
    site_name: str | None = None
    # Publication time as the provider reports it (RFC 3339 for DeepSeek).
    date: str | None = None


class SearchProvider(ABC):
    """One web-search backend."""

    # Catalog id ([web_search] provider = "<id>").
    id: str
    # Upper bound for ``count`` the backend accepts.
    max_count: int = 10
    # Extra text appended to the tool description (e.g. query operators).
    query_help: str = ""

    @abstractmethod
    def search(self, query: str, count: int) -> list[SearchResult]:
        """Run ``query``; raise :class:`SearchError` on failure."""
