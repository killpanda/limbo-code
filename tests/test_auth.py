"""Stored logins (~/.limbo/auth.json), key resolution order and the auth CLI."""

from __future__ import annotations

import io
import stat

import pytest

from limbo import auth, cli_auth
from limbo.config import Config, ProviderOverride
from limbo.llm import catalog
from limbo.llm.catalog import (
    api_key_source,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    thinking_choices,
)


def test_save_load_remove_and_permissions():
    auth.save("glm", " key-123 ", base_url="https://relay.example/v4")
    path = auth.auth_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    cred = auth.get("glm")
    assert cred == auth.Credential("key-123", "https://relay.example/v4")
    assert auth.remove("glm")
    assert not auth.remove("glm")
    assert auth.get("glm") is None


def test_corrupt_file_is_ignored_with_warning():
    path = auth.auth_path()
    path.write_text("{nope")
    with pytest.warns(UserWarning):
        assert auth.load_all() == {}


def test_mask():
    assert auth.mask("sk-abcdefghijklmnop") == "sk-ab…mnop"
    assert auth.mask("short") == "•••••"


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


def test_thinking_choices():
    assert thinking_choices(resolve_model("unknown-model")) == []
    kimi = next(s for s in catalog.CATALOG.values() if s.provider.id == "kimi-coding")
    assert thinking_choices(kimi) == [None, "low", "high", "max"]


def test_split_model_arg():
    assert cli_auth.split_model_arg("kimi-k3:high") == ("kimi-k3", "high")
    assert cli_auth.split_model_arg("glm-4.7") == ("glm-4.7", None)
    assert cli_auth.split_model_arg("qwen3:32b") == ("qwen3:32b", None)


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


def test_login_picks_provider_interactively():
    stdin = io.StringIO("glm\nsk-from-stdin\n")
    rc = cli_auth.login(None, verify=False, stdin=stdin, stdout=io.StringIO())
    assert rc == 0
    assert auth.get("glm").key == "sk-from-stdin"


def test_login_rejects_unknown_provider():
    assert cli_auth.login("nope", key="x", verify=False) == 1


def test_login_stops_on_invalid_key(monkeypatch):
    from limbo.llm import verify

    async def fake_check(provider, key, base_url=None, timeout=8.0):
        return verify.KeyCheck(verify.INVALID, "拒绝")

    monkeypatch.setattr(cli_auth, "check_key", fake_check)
    assert cli_auth.login("glm", key="bad", stdout=io.StringIO()) == 1
    assert auth.get("glm") is None


def test_models_listing_marks_current():
    out = io.StringIO()
    config = Config()
    cli_auth.list_models(config, stdout=out)
    assert f"● {config.llm.model}" in out.getvalue()
