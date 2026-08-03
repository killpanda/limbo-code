"""Tests for the provider/model catalog and client factory."""

import pytest

from limbo.config import Config, ProviderOverride
from limbo.llm.anthropic_client import AnthropicMessagesClient
from limbo.llm.catalog import (
    DEEPSEEK,
    DEFAULT_BASE_URL,
    GENERIC_OPENAI,
    GLM_CODING,
    KIMI_CODING,
    MOONSHOT,
    ModelSpec,
    ProviderSpec,
    resolve_api_key,
    resolve_base_url,
    resolve_headers,
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


def test_deepseek_v4_specs():
    pro = resolve_model("deepseek-v4-pro")
    assert pro.provider is DEEPSEEK
    assert pro.reasoning is True
    assert pro.thinking_format == "deepseek"
    assert pro.max_tokens == 65_536

    flash = resolve_model("deepseek-v4-flash")
    assert flash.provider is DEEPSEEK
    assert flash.reasoning is True
    assert flash.thinking_format == "deepseek"


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


def _config(**kwargs) -> Config:
    """Build a Config with the given llm/providers fields pre-set."""
    cfg = Config()
    for key, value in kwargs.items():
        if key == "providers":
            cfg.providers = value
        else:
            setattr(cfg.llm, key, value)
    return cfg


def test_base_url_follows_provider_when_config_left_at_default():
    spec = resolve_model("kimi-k3")
    assert resolve_base_url(spec, Config()) == "https://api.moonshot.ai/v1"


def test_configured_base_url_always_wins():
    spec = resolve_model("kimi-k3")
    cfg = _config(base_url="https://proxy.example.com/v1")
    assert resolve_base_url(spec, cfg) == "https://proxy.example.com/v1"


def test_provider_override_base_url_beats_explicit_llm_base_url():
    # [providers.<id>] is more specific than the global [llm] setting and
    # wins even over an explicitly configured [llm] base_url (documented
    # exception to "explicit always wins").
    spec = resolve_model("glm-4.7")
    cfg = _config(
        base_url="https://proxy.example.com/v1",
        providers={
            "glm": ProviderOverride(base_url="https://relay.example.com/v4")
        },
    )
    assert resolve_base_url(spec, cfg) == "https://relay.example.com/v4"


def test_base_url_default_for_unknown_model():
    spec = resolve_model("my-local-model")
    assert resolve_base_url(spec, Config()) == DEFAULT_BASE_URL


def test_api_key_prefers_config(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    spec = resolve_model("kimi-k3")
    assert resolve_api_key(spec, _config(api_key="config-key")) == "config-key"


def test_api_key_falls_back_to_provider_env(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-key")
    spec = resolve_model("kimi-k3")
    assert resolve_api_key(spec, Config()) == "env-key"


def test_api_key_none_without_env(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    spec = resolve_model("kimi-k3")
    assert resolve_api_key(spec, Config()) is None


def test_provider_override_api_key_beats_llm_and_env(monkeypatch):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "env-key")
    spec = resolve_model("glm-4.7")
    cfg = _config(
        api_key="llm-key",
        providers={"glm": ProviderOverride(api_key="override-key")},
    )
    assert resolve_api_key(spec, cfg) == "override-key"


def test_provider_override_api_key_env_renames_env_var(monkeypatch):
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    monkeypatch.setenv("MY_GLM_KEY", "renamed-env-key")
    spec = resolve_model("glm-4.7")
    cfg = _config(providers={"glm": ProviderOverride(api_key_env="MY_GLM_KEY")})
    assert resolve_api_key(spec, cfg) == "renamed-env-key"


def test_resolve_headers_merges_provider_and_override():
    spec = resolve_model("k3")
    assert resolve_headers(spec, Config()) == {"User-Agent": "KimiCLI/1.5"}
    cfg = _config(
        providers={
            "kimi-coding": ProviderOverride(
                headers={"User-Agent": "custom/1.0", "x-relay": "on"}
            )
        }
    )
    assert resolve_headers(spec, cfg) == {
        "User-Agent": "custom/1.0",
        "x-relay": "on",
    }


def test_glm_coding_plan_catalog():
    spec = resolve_model("glm-4.7")
    assert spec.provider is GLM_CODING
    assert spec.provider.api == "openai-completions"
    assert spec.provider.base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert spec.provider.api_key_env == "ZHIPUAI_API_KEY"
    assert spec.provider.tool_extra_body == {"tool_stream": True}
    assert spec.context_window == 204_800
    assert spec.max_tokens == 131_072
    assert spec.reasoning is True
    assert spec.thinking_format == "zai"
    assert spec.thinking_levels == {}


def test_glm_model_whitelist():
    # Coding Plan subscription whitelist (mirrors pi's zai-coding-cn.json).
    expected = {
        "glm-4.5-air": (131_072, 98_304),
        "glm-4.7": (204_800, 131_072),
        "glm-5-turbo": (200_000, 131_072),
        "glm-5.1": (200_000, 131_072),
        "glm-5.2": (1_000_000, 131_072),
        "glm-5v-turbo": (200_000, 131_072),
    }
    for model_id, (ctx, max_tokens) in expected.items():
        spec = resolve_model(model_id)
        assert spec.provider is GLM_CODING, model_id
        assert spec.context_window == ctx, model_id
        assert spec.max_tokens == max_tokens, model_id
        assert spec.thinking_format == "zai", model_id
    assert resolve_model("glm-5v-turbo").vision is True
    assert resolve_model("glm-4.7").vision is False


def test_glm_5_2_reasoning_effort_levels():
    spec = resolve_model("glm-5.2")
    # pi's thinkingLevelMap: minimal unsupported, low/medium clamp to high.
    assert spec.thinking_levels == {"low": "high", "high": "high", "max": "max"}


def test_glm_base_url_follows_coding_endpoint_by_default():
    spec = resolve_model("glm-4.7")
    assert resolve_base_url(spec, Config()) == (
        "https://open.bigmodel.cn/api/coding/paas/v4"
    )


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


def test_factory_creates_responses_client_for_codex():
    from limbo.llm.responses_client import OpenAIResponsesClient

    cfg = Config()
    cfg.llm.model = "gpt-5.5"
    cfg.providers["codex"] = ProviderOverride(api_key="relay-key")
    client = create_llm_client(cfg)
    assert isinstance(client, OpenAIResponsesClient)
    assert client.spec.provider.id == "codex"
    assert client.spec.provider.api == "openai-responses"


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


def test_vision_defaults_to_false():
    # Text-only catalog entries stay non-vision (conservative default).
    assert resolve_model("deepseek-v4-flash").vision is False
    assert resolve_model("kimi-k3").vision is False


def test_kimi_k2_5_is_vision_capable():
    assert resolve_model("kimi-k2.5").vision is True


def test_unknown_model_vision_defaults_to_false():
    # GENERIC_OPENAI fallback: image attachments degrade to path references
    # rather than breaking an unrecognized endpoint with base64 payloads.
    assert resolve_model("my-local-model").vision is False
