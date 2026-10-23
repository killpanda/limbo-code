"""DeepSeek web search (https://api-docs.deepseek.com/guides/search)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from limbo.websearch.base import SearchError, SearchProvider, SearchResult

if TYPE_CHECKING:
    from limbo.config import Config

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

# Statuses the API documents, mapped to a message the model can act on.
_STATUS_HINTS = {
    401: "missing or invalid API key",
    402: "insufficient DeepSeek account balance",
    408: "request timed out; retry later",
    429: "rate limit exceeded; retry later",
    504: "upstream timed out; retry later",
}


class DeepSeekSearch(SearchProvider):
    id = "deepseek"
    max_count = 50
    query_help = (
        " Query operators: site:example.com (combine with OR), "
        "-site:example.com, -term, \"exact phrase\", lang:zh,en, "
        "after:2025-01-01, before:2026."
    )

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/tools/search"
        self._timeout = timeout
        self._transport = transport

    def search(self, query: str, count: int) -> list[SearchResult]:
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"query": query, "count": count},
                )
        except httpx.HTTPError as e:
            raise SearchError(f"DeepSeek search request failed: {e}") from e
        if response.status_code != 200:
            raise SearchError(_error_message(response))
        try:
            results = response.json().get("results") or []
        except (ValueError, AttributeError) as e:
            raise SearchError(f"DeepSeek search returned invalid JSON: {e}") from e
        return [_to_result(item) for item in results if isinstance(item, dict)]


def _error_message(response: httpx.Response) -> str:
    detail = ""
    try:
        error = response.json().get("error")
        if isinstance(error, dict):
            detail = str(error.get("message") or "")
    except (ValueError, AttributeError):
        pass
    hint = _STATUS_HINTS.get(response.status_code, "")
    parts = [p for p in (hint, detail) if p]
    suffix = f": {'; '.join(parts)}" if parts else ""
    return f"DeepSeek search failed (HTTP {response.status_code}){suffix}"


def _to_result(item: dict[str, Any]) -> SearchResult:
    def text(key: str) -> str:
        value = item.get(key)
        return value.strip() if isinstance(value, str) else ""

    return SearchResult(
        title=text("title"),
        url=text("url"),
        summary=text("summary") or text("snippet"),
        site_name=text("site_name") or None,
        date=text("date") or None,
    )


def resolve_api_key(config: Config) -> str | None:
    """The key for DeepSeek search.

    ``[web_search] api_key`` first; otherwise inherit the DeepSeek model
    provider's key with the usual resolution (``[providers.deepseek]
    api_key`` → ``limbo auth login`` → ``[llm] api_key`` → ``$DEEPSEEK_API_KEY``),
    except that ``[llm] api_key`` only counts while the active model is a
    DeepSeek one — otherwise it belongs to another provider and must not
    be sent to DeepSeek.
    """
    from limbo.llm.catalog import resolve_api_key as resolve_model_key
    from limbo.llm.catalog import resolve_model

    if config.web_search.api_key:
        return config.web_search.api_key
    active = resolve_model(config.llm.model)
    if active.provider.id == "deepseek":
        return resolve_model_key(active, config)
    scoped = config.model_copy(
        update={"llm": config.llm.model_copy(update={"api_key": None})}
    )
    return resolve_model_key(resolve_model("deepseek-v4-pro"), scoped)


def build(config: Config) -> DeepSeekSearch | None:
    key = resolve_api_key(config)
    if not key:
        return None
    settings = config.web_search
    return DeepSeekSearch(key, base_url=settings.base_url, timeout=settings.timeout)
