"""Tests for the provider/model catalog and client factory."""

import pytest

from limbo.config import Config
from limbo.llm.anthropic_client import AnthropicMessagesClient
from limbo.llm.catalog import (
    DEFAULT_BASE_URL,
    GENERIC_OPENAI,
    KIMI_CODING,
    MOONSHOT,
    ModelSpec,
    ProviderSpec,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from limbo.llm.factory import create_llm_client, register_client
from limbo.llm.openai_client import OpenAICompatibleClient


def test_kimi_k3_spec():
    spec = resolve_model("kimi-k3")
    assert spec.provider is MOONSHOT
    assert spec.provider.api == "openai-completions"
    assert spec.provider.base_url == "https://api.moonshot.ai/v1"
    assert spec.provider.api_key_env == "MOONSHOT_API_KEY"
    assert spec.context_window == 1_048_576
    assert spec.max_tokens == 131_072
    assert spec.reasoning is True
    assert spec.thinking_format == "openai"
    assert spec.thinking_levels == {"low": "low", "high": "high", "max": "max"}
    assert spec.thinking_can_disable is False
    assert spec.requires_reasoning_content is True


def test_kimi_k2_thinking_spec():
    spec = resolve_model("kimi-k2-thinking")
    assert spec.reasoning is True
    assert spec.thinking_format == "deepseek"
    assert spec.context_window == 262_144
    assert spec.requires_reasoning_content is False


def test_unknown_model_falls_back_to_generic_provider():
    spec = resolve_model("my-local-model")
    assert spec.provider is GENERIC_OPENAI
    assert spec.provider.api == "openai-completions"
    assert spec.context_window == 128_000
    assert spec.reasoning is False


def test_base_url_follows_provider_when_config_left_at_default():
    spec = resolve_model("kimi-k3")
    assert (
        resolve_base_url(spec, DEFAULT_BASE_URL) == "https://api.moonshot.ai/v1"
    )


def test_configured_base_url_always_wins():
    spec = resolve_model("kimi-k3")
    assert resolve_base_url(spec, "https://proxy.example.com/v1") == (
        "https://proxy.example.com/v1"
    )


def test_base_url_default_for_unknown_model():
    spec = resolve_model("my-local-model")
    assert resolve_base_url(spec, DEFAULT_BASE_URL) == DEFAULT_BASE_URL


def test_api_key_prefers_config(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    spec = resolve_model("kimi-k3")
    assert resolve_api_key(spec, "config-key") == "config-key"


def test_api_key_falls_back_to_provider_env(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    spec = resolve_model("kimi-k3")
    assert resolve_api_key(spec, None) == "env-key"


def test_api_key_none_without_env(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    spec = resolve_model("kimi-k3")
    assert resolve_api_key(spec, None) is None


def test_factory_creates_openai_client_for_kimi():
    cfg = Config()
    cfg.llm.model = "kimi-k3"
    client = create_llm_client(cfg)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.spec.id == "kimi-k3"


def test_kimi_coding_k3_spec():
    spec = resolve_model("k3")
    assert spec.provider is KIMI_CODING
    assert spec.provider.api == "anthropic-messages"
    assert spec.provider.base_url == "https://api.kimi.com/coding"
    assert spec.provider.api_key_env == "KIMI_API_KEY"
    assert spec.provider.headers == {"User-Agent": "KimiCLI/1.5"}
    assert spec.context_window == 1_048_576
    assert spec.max_tokens == 131_072
    assert spec.reasoning is True
    assert spec.thinking_format == "anthropic-adaptive"
    assert spec.thinking_levels == {"low": "low", "high": "high", "max": "max"}
    assert spec.thinking_can_disable is False


def test_factory_creates_anthropic_client_for_kimi_coding():
    cfg = Config()
    cfg.llm.model = "k3"
    cfg.llm.api_key = "sk-kim-test"
    client = create_llm_client(cfg)
    assert isinstance(client, AnthropicMessagesClient)
    assert client.spec.provider.id == "kimi-coding"


def test_factory_rejects_unsupported_api_dialect(monkeypatch):
    import limbo.llm.catalog as catalog

    google_provider = ProviderSpec(
        id="google",
        api="google-generative-ai",
        base_url="https://generativelanguage.googleapis.com",
    )
    monkeypatch.setitem(
        catalog.CATALOG,
        "fake-gemini",
        ModelSpec(id="fake-gemini", provider=google_provider),
    )
    cfg = Config()
    cfg.llm.model = "fake-gemini"
    with pytest.raises(ValueError, match="google-generative-ai"):
        create_llm_client(cfg)


def test_register_client_adds_dialect():
    import limbo.llm.catalog as catalog
    import limbo.llm.factory as factory_module

    class DummyClient:
        def __init__(self, config):
            self.config = config

    register_client("dummy-api", DummyClient)
    catalog.CATALOG["dummy-model"] = ModelSpec(
        id="dummy-model",
        provider=ProviderSpec(id="dummy", api="dummy-api"),
    )
    try:
        cfg = Config()
        cfg.llm.model = "dummy-model"
        assert isinstance(create_llm_client(cfg), DummyClient)
    finally:
        factory_module._FACTORIES.pop("dummy-api", None)
        catalog.CATALOG.pop("dummy-model", None)
