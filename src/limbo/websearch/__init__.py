"""Pluggable web search: provider catalog + factory.

``[web_search] provider`` picks a backend from :data:`PROVIDERS`; each
entry builds a configured :class:`SearchProvider` from the config, or
returns None when it has no credentials (the tool is then not offered).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from limbo.websearch import deepseek
from limbo.websearch.base import SearchError, SearchProvider, SearchResult

if TYPE_CHECKING:
    from limbo.config import Config

__all__ = ["PROVIDERS", "SearchError", "SearchProvider", "SearchResult", "build_provider"]

PROVIDERS: dict[str, Callable[[Config], SearchProvider | None]] = {
    "deepseek": deepseek.build,
}


def build_provider(config: Config) -> SearchProvider | None:
    """The configured provider, or None (disabled, unknown, or no key)."""
    settings = config.web_search
    if not settings.enabled:
        return None
    factory = PROVIDERS.get(settings.provider)
    return factory(config) if factory else None
