import json
from pathlib import Path

import httpx
import pytest

from limbo import auth
from limbo.config import Config
from limbo.tools.registry import ToolRegistry
from limbo.tools.web_search import WebSearchTool
from limbo.websearch import SearchError, SearchProvider, SearchResult, build_provider
from limbo.websearch.deepseek import DeepSeekSearch, resolve_api_key


class FakeProvider(SearchProvider):
    id = "fake"
    max_count = 5

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query, count):
        self.calls.append((query, count))
        if self.error:
            raise SearchError(self.error)
        return self.results


def make_tool(tmp_path, provider, count=3):
    return WebSearchTool(Path(tmp_path), provider, count)


# -- tool -------------------------------------------------------------------


def test_formats_results(tmp_path):
    provider = FakeProvider([
        SearchResult("Title A", "https://a.example", "Summary A", "A site", "2026-01-01T00:00:00Z"),
        SearchResult("", "https://b.example", ""),
    ])
    result = make_tool(tmp_path, provider).execute({"query": "  limbo  "})
    assert result.success
    assert provider.calls == [("limbo", 3)]
    assert result.output == (
        "[1] Title A\nhttps://a.example\nA site · 2026-01-01T00:00:00Z\nSummary A"
        "\n\n[2] https://b.example\nhttps://b.example"
    )


def test_count_is_clamped_to_provider_max(tmp_path):
    provider = FakeProvider()
    tool = make_tool(tmp_path, provider, count=99)
    assert tool.default_count == 5
    tool.execute({"query": "q", "count": 40})
    tool.execute({"query": "q", "count": 0})
    assert [c for _, c in provider.calls] == [5, 1]


def test_no_results(tmp_path):
    result = make_tool(tmp_path, FakeProvider()).execute({"query": "nothing"})
    assert result.success and result.output == "No results for: nothing"


@pytest.mark.parametrize(
    "args", [{}, {"query": "  "}, {"query": "x" * 129}, {"query": "q", "count": "3"}]
)
def test_bad_arguments(tmp_path, args):
    result = make_tool(tmp_path, FakeProvider()).execute(args)
    assert not result.success


def test_provider_error_becomes_tool_error(tmp_path):
    result = make_tool(tmp_path, FakeProvider(error="boom")).execute({"query": "q"})
    assert not result.success and result.error == "boom"


# -- DeepSeek provider ---------------------------------------------------------


def deepseek(handler):
    return DeepSeekSearch("sk-test", transport=httpx.MockTransport(handler))


def test_deepseek_request_and_parse():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{
            "title": "T", "url": "https://x", "snippet": "short", "summary": "long",
            "site_name": None, "date": "2026-06-01T00:00:00+00:00", "authority": 3,
        }, {"title": "U", "url": "https://y", "snippet": "only snippet", "summary": ""}]})

    results = deepseek(handler).search("q", 7)
    assert seen == {
        "url": "https://api.deepseek.com/v1/tools/search",
        "auth": "Bearer sk-test",
        "body": {"query": "q", "count": 7},
    }
    assert results[0] == SearchResult("T", "https://x", "long", None, "2026-06-01T00:00:00+00:00")
    assert results[1].summary == "only snippet"


def test_deepseek_http_error_message():
    def handler(request):
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    with pytest.raises(SearchError, match=r"HTTP 402.*balance.*Insufficient Balance"):
        deepseek(handler).search("q", 1)


def test_deepseek_transport_error():
    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(SearchError, match="request failed"):
        deepseek(handler).search("q", 1)


def test_deepseek_base_url_override():
    provider = DeepSeekSearch("k", base_url="https://relay.example/v1/")
    assert provider._url == "https://relay.example/v1/tools/search"


# -- key resolution / registration --------------------------------------------


def config(**data):
    return Config.model_validate(data)


def test_own_key_wins(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env")
    assert resolve_api_key(config(web_search={"api_key": "own"})) == "own"


def test_inherits_deepseek_provider_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env")
    assert resolve_api_key(config()) == "env"
    cfg = config(providers={"deepseek": {"api_key": "prov"}}, llm={"model": "kimi-k2.5"})
    assert resolve_api_key(cfg) == "prov"


def test_inherits_auth_login():
    auth.save("deepseek", "stored")
    assert resolve_api_key(config(llm={"model": "kimi-k2.5"})) == "stored"


def test_llm_key_only_inherited_for_deepseek_model():
    assert resolve_api_key(config(llm={"api_key": "llm"})) == "llm"
    assert resolve_api_key(config(llm={"api_key": "llm", "model": "kimi-k2.5"})) is None


def test_registered_only_with_key(tmp_path):
    assert "web_search" not in ToolRegistry(tmp_path, config()).names()
    cfg = config(web_search={"api_key": "k", "count": 4})
    reg = ToolRegistry(tmp_path, cfg)
    tool = reg.get("web_search")
    assert isinstance(tool, WebSearchTool) and tool.default_count == 4
    assert reg.replay_policy("web_search") == "safe"
    disabled = config(web_search={"api_key": "k", "enabled": False})
    assert "web_search" not in ToolRegistry(tmp_path, disabled).names()


def test_unknown_provider_is_not_built():
    assert build_provider(config(web_search={"api_key": "k", "provider": "nope"})) is None
