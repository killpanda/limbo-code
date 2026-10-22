"""Tests for the model-switch module (/model domain logic, no Textual)."""

from __future__ import annotations

import pytest

from limbo.config import Config
from limbo.model_switch import (
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


# -- reload_llm_config ---------------------------------------------------------


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


