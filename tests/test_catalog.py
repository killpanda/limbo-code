"""Tests for the provider/model catalog and client factory."""


from limbo.config import Config, ProviderOverride
from limbo.llm.catalog import (
    resolve_api_key,
    resolve_model,
)


def _config(**kwargs) -> Config:
    """Build a Config with the given llm/providers fields pre-set."""
    cfg = Config()
    for key, value in kwargs.items():
        if key == "providers":
            cfg.providers = value
        else:
            setattr(cfg.llm, key, value)
    return cfg


def test_provider_override_api_key_beats_llm_and_env(monkeypatch):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "env-key")
    spec = resolve_model("glm-4.7")
    cfg = _config(
        api_key="llm-key",
        providers={"glm": ProviderOverride(api_key="override-key")},
    )
    assert resolve_api_key(spec, cfg) == "override-key"


