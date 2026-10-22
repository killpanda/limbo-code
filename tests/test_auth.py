"""Stored logins (~/.limbo/auth.json), key resolution order and the auth CLI."""

from __future__ import annotations

import io

from limbo import auth, cli_auth
from limbo.config import Config, ProviderOverride
from limbo.llm.catalog import (
    api_key_source,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)


def test_resolution_order(monkeypatch):
    spec = resolve_model("glm-4.7")
    config = Config()
    monkeypatch.setenv("ZHIPUAI_API_KEY", "from-env")
    assert resolve_api_key(spec, config) == "from-env"
    assert api_key_source(spec, config) == "$ZHIPUAI_API_KEY"
    config.llm.api_key = "global"
    assert resolve_api_key(spec, config) == "global"
    auth.save("glm", "stored", base_url="https://relay.example/v4")
    # A stored login beats the global [llm] key …
    assert resolve_api_key(spec, config) == "stored"
    assert api_key_source(spec, config) == "auth.json"
    assert resolve_base_url(spec, config) == "https://relay.example/v4"
    # … but a [providers.<id>] key in config.toml is still the most specific.
    config.providers["glm"] = ProviderOverride(api_key="override")
    assert resolve_api_key(spec, config) == "override"


def test_login_list_logout_cli(monkeypatch):
    out = io.StringIO()
    rc = cli_auth.login("glm", key="sk-test-1234567890", verify=False, stdout=out)
    assert rc == 0
    assert "sk-te…7890" in out.getvalue()
    out = io.StringIO()
    cli_auth.list_logins(Config(), stdout=out)
    assert "auth.json" in out.getvalue()
    assert cli_auth.logout("glm", stdout=io.StringIO()) == 0
    assert auth.get("glm") is None


