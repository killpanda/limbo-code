"""Tests for the model-switch module (/model domain logic, no Textual)."""

from __future__ import annotations

import pytest

from limbo.config import Config
from limbo.model_switch import (
    prepare_model_switch,
    reload_llm_config,
    swap_llm_client,
)


def _config(**llm_overrides) -> Config:
    config = Config()
    config.llm.api_key = "sk-test"
    for key, value in llm_overrides.items():
        setattr(config.llm, key, value)
    return config


class _StubAgent:
    def __init__(self) -> None:
        self.updated_with: list = []

    def update_llm(self, client) -> None:
        self.updated_with.append(client)


class _CloseableClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


# -- prepare_model_switch ------------------------------------------------------


def test_prepare_same_model_is_a_noop():
    config = _config(model="deepseek-v4-pro")
    verdict = prepare_model_switch("deepseek-v4-pro", config)
    assert not verdict.switched
    assert any("当前已是" in notice for notice in verdict.notices)


def test_prepare_missing_api_key_refuses():
    config = _config()
    config.llm.api_key = None
    verdict = prepare_model_switch("kimi-k3", config)
    assert not verdict.switched
    assert any("MOONSHOT_API_KEY" in notice for notice in verdict.notices)
    assert config.llm.model != "kimi-k3"


def test_prepare_resets_incompatible_thinking_effort():
    config = _config(model="deepseek-v4-pro", thinking_effort="medium")
    verdict = prepare_model_switch("kimi-k3", config)
    assert verdict.switched
    assert config.llm.thinking_effort is None
    assert any("thinking_effort" in notice for notice in verdict.notices)


def test_prepare_keeps_supported_thinking_effort():
    config = _config(model="deepseek-v4-pro", thinking_effort="high")
    verdict = prepare_model_switch("kimi-k3", config)
    assert verdict.switched
    assert config.llm.thinking_effort == "high"


def test_prepare_unknown_model_warns_generic():
    config = _config(model="deepseek-v4-pro")
    verdict = prepare_model_switch("some-custom-model", config)
    assert verdict.switched
    assert any("未知模型" in notice for notice in verdict.notices)


def test_prepare_sets_config_model_last():
    config = _config(model="deepseek-v4-pro")
    verdict = prepare_model_switch("glm-4.7", config)
    assert verdict.switched
    assert config.llm.model == "glm-4.7"


# -- reload_llm_config ---------------------------------------------------------


def test_reload_llm_config_replaces_llm_and_providers(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[llm]\nmodel = "glm-4.7"\n\n[providers.glm]\napi_key = "k"\n'
    )
    monkeypatch.setattr("limbo.config.DEFAULT_CONFIG_PATH", config_file)
    config = Config()
    reload_llm_config(config)
    assert config.llm.model == "glm-4.7"
    assert config.providers["glm"].api_key == "k"


# -- swap_llm_client -----------------------------------------------------------


@pytest.mark.asyncio
async def test_swap_closes_old_client_and_updates_agent():
    config = _config(model="deepseek-v4-pro")
    old = _CloseableClient()
    agent = _StubAgent()
    new_client, notices = await swap_llm_client(config, old, agent)
    assert old.closed
    assert agent.updated_with == [new_client]
    assert any("已切换模型 deepseek-v4-pro (deepseek)" in n for n in notices)


@pytest.mark.asyncio
async def test_swap_tolerates_client_without_close():
    config = _config(model="deepseek-v4-pro")
    agent = _StubAgent()
    new_client, _ = await swap_llm_client(config, object(), agent)
    assert agent.updated_with == [new_client]


@pytest.mark.asyncio
async def test_swap_persists_model_to_config(tmp_path, monkeypatch):
    """The switch survives restarts: the new model is written to config.toml
    (isolated here — the real one belongs to the developer)."""
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr("limbo.config.DEFAULT_CONFIG_PATH", config_file)
    config = _config(model="glm-4.7")
    await swap_llm_client(config, _CloseableClient(), _StubAgent())
    assert 'model = "glm-4.7"' in config_file.read_text()


@pytest.mark.asyncio
async def test_swap_reports_persistence_failure(monkeypatch):
    monkeypatch.setattr("limbo.model_switch.save_model_to_config", lambda model: False)
    config = _config(model="deepseek-v4-pro")
    _, notices = await swap_llm_client(config, _CloseableClient(), _StubAgent())
    assert any("配置写回失败" in notice for notice in notices)
