"""LLM client factory.

Selects the client implementation for the model's provider API dialect
(``ProviderSpec.api``). New dialects (e.g. ``anthropic-messages`` for Kimi's
coding endpoint) plug in via ``register_client`` without touching call sites.
"""

from __future__ import annotations

from collections.abc import Callable

from limbo.config import Config
from limbo.llm.anthropic_client import AnthropicMessagesClient
from limbo.llm.catalog import (
    API_ANTHROPIC_MESSAGES,
    API_OPENAI_COMPLETIONS,
    resolve_model,
)
from limbo.llm.client import LLMClient
from limbo.llm.openai_client import OpenAICompatibleClient

ClientFactory = Callable[[Config], LLMClient]

_FACTORIES: dict[str, ClientFactory] = {
    API_OPENAI_COMPLETIONS: OpenAICompatibleClient,
    API_ANTHROPIC_MESSAGES: AnthropicMessagesClient,
}


def register_client(api: str, factory: ClientFactory) -> None:
    """Register a client factory for a provider API dialect."""
    _FACTORIES[api] = factory


def create_llm_client(config: Config) -> LLMClient:
    """Create the LLM client for the configured model's provider."""
    spec = resolve_model(config.llm.model)
    factory = _FACTORIES.get(spec.provider.api)
    if factory is None:
        supported = ", ".join(sorted(_FACTORIES))
        raise ValueError(
            f"No LLM client implementation for API dialect "
            f"{spec.provider.api!r} (model {spec.id!r}, provider "
            f"{spec.provider.id!r}). Supported dialects: {supported}."
        )
    return factory(config)
