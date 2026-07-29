"""Provider/model catalog.

Follows pi's two-level metadata approach (built-in ``providers/data/*.json``
plus ``~/.pi/agent/models.json``):

- A **provider** owns the API dialect (``api``), endpoint (``base_url``), and
  credential source (``api_key_env``). The ``api`` value decides which LLM
  client implementation speaks to it (see ``limbo.llm.factory``).
- A **model** belongs to a provider and carries its own API characteristics —
  context window, max output tokens, reasoning capability, thinking-parameter
  format, and compatibility quirks — so the client adapts request/response
  handling per model instead of assuming one universal dialect.

Unknown models fall back to a generic OpenAI-compatible provider with
conservative defaults, so any OpenAI-compatible endpoint still works via
``[llm] base_url`` + ``model``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_TOKENS = 16_384

# API dialects. Only "openai-completions" has a client implementation today;
# the factory raises a clear error for the rest until one is added.
API_OPENAI_COMPLETIONS = "openai-completions"
API_ANTHROPIC_MESSAGES = "anthropic-messages"

# Mainland-China Moonshot endpoint; select via an explicit [llm] base_url
# override (same API and model ids as the international endpoint).
MOONSHOT_CN_BASE_URL = "https://api.moonshot.cn/v1"


@dataclass(frozen=True)
class ProviderSpec:
    """An API provider: dialect + endpoint + credential source."""

    id: str
    api: str
    base_url: str = ""
    api_key_env: str | None = None
    # Extra headers sent on every request (e.g. Kimi For Coding requires a
    # specific User-Agent).
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Per-model API characteristics."""

    id: str
    provider: ProviderSpec
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning: bool = False
    # How thinking is controlled: "openai" (reasoning_effort parameter),
    # "deepseek" (thinking: {type: enabled|disabled}), or None (the model
    # reasons but the API exposes no switch, e.g. deepseek-reasoner).
    thinking_format: str | None = None
    # Supported thinking levels mapped to provider values (openai format).
    # A level absent from the map is unsupported by the model.
    thinking_levels: dict[str, str] = field(default_factory=dict)
    # Whether thinking can be turned off at all (deepseek format).
    thinking_can_disable: bool = True
    # Replay reasoning_content (empty string when absent) on assistant
    # messages. Kimi K3 rejects tool-call replays without the field.
    requires_reasoning_content: bool = False


DEEPSEEK = ProviderSpec(
    id="deepseek",
    api=API_OPENAI_COMPLETIONS,
    base_url=DEFAULT_BASE_URL,
    api_key_env="DEEPSEEK_API_KEY",
)

MOONSHOT = ProviderSpec(
    id="moonshotai",
    api=API_OPENAI_COMPLETIONS,
    base_url="https://api.moonshot.ai/v1",
    api_key_env="MOONSHOT_API_KEY",
)

# Fallback for models not in the catalog.
GENERIC_OPENAI = ProviderSpec(id="custom", api=API_OPENAI_COMPLETIONS)

# Kimi For Coding (subscription): Anthropic Messages dialect. Mirrors pi's
# kimi-coding provider (providers/data/kimi-coding.json).
KIMI_CODING = ProviderSpec(
    id="kimi-coding",
    api=API_ANTHROPIC_MESSAGES,
    base_url="https://api.kimi.com/coding",
    api_key_env="KIMI_API_KEY",
    headers={"User-Agent": "KimiCLI/1.5"},
)


def _kimi_coding(
    model_id: str,
    *,
    context_window: int = 262_144,
    max_tokens: int = 32_768,
) -> ModelSpec:
    # All Kimi For Coding models reason with Anthropic adaptive thinking
    # (thinking: {type: adaptive} + output_config.effort); thinking cannot
    # be disabled. Replayed thinking blocks may carry an empty signature.
    return ModelSpec(
        id=model_id,
        provider=KIMI_CODING,
        context_window=context_window,
        max_tokens=max_tokens,
        reasoning=True,
        thinking_format="anthropic-adaptive",
        thinking_levels={"low": "low", "high": "high", "max": "max"},
        thinking_can_disable=False,
    )


def _moonshot(
    model_id: str,
    *,
    context_window: int = 262_144,
    max_tokens: int = 262_144,
    reasoning: bool = False,
    thinking_format: str | None = None,
    thinking_levels: dict[str, str] | None = None,
    thinking_can_disable: bool = True,
    requires_reasoning_content: bool = False,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider=MOONSHOT,
        context_window=context_window,
        max_tokens=max_tokens,
        reasoning=reasoning,
        thinking_format=thinking_format,
        thinking_levels=thinking_levels or {},
        thinking_can_disable=thinking_can_disable,
        requires_reasoning_content=requires_reasoning_content,
    )


# Values mirror pi's built-in moonshotai catalog (providers/data/moonshotai.json).
CATALOG: dict[str, ModelSpec] = {
    "deepseek-chat": ModelSpec(
        id="deepseek-chat", provider=DEEPSEEK, max_tokens=8_192
    ),
    "deepseek-reasoner": ModelSpec(
        id="deepseek-reasoner",
        provider=DEEPSEEK,
        max_tokens=65_536,
        reasoning=True,
    ),
    # -- Kimi (Moonshot AI) -------------------------------------------------
    "kimi-k2-0711-preview": _moonshot(
        "kimi-k2-0711-preview", context_window=131_072, max_tokens=16_384
    ),
    "kimi-k2-0905-preview": _moonshot("kimi-k2-0905-preview"),
    "kimi-k2-turbo-preview": _moonshot("kimi-k2-turbo-preview"),
    "kimi-k2-thinking": _moonshot(
        "kimi-k2-thinking", reasoning=True, thinking_format="deepseek"
    ),
    "kimi-k2-thinking-turbo": _moonshot(
        "kimi-k2-thinking-turbo", reasoning=True, thinking_format="deepseek"
    ),
    "kimi-k2.5": _moonshot(
        "kimi-k2.5", reasoning=True, thinking_format="deepseek"
    ),
    "kimi-k2.6": _moonshot(
        "kimi-k2.6", reasoning=True, thinking_format="deepseek"
    ),
    "kimi-k2.7-code": _moonshot(
        "kimi-k2.7-code",
        reasoning=True,
        thinking_format="deepseek",
        thinking_can_disable=False,
    ),
    "kimi-k2.7-code-highspeed": _moonshot(
        "kimi-k2.7-code-highspeed",
        reasoning=True,
        thinking_format="deepseek",
        thinking_can_disable=False,
    ),
    # Kimi K3: 1M context, OpenAI-style reasoning_effort, thinking cannot be
    # disabled, and assistant replays must carry reasoning_content.
    "kimi-k3": _moonshot(
        "kimi-k3",
        context_window=1_048_576,
        max_tokens=131_072,
        reasoning=True,
        thinking_format="openai",
        thinking_levels={"low": "low", "high": "high", "max": "max"},
        thinking_can_disable=False,
        requires_reasoning_content=True,
    ),
    # -- Kimi For Coding (Anthropic Messages dialect) ------------------------
    "k3": _kimi_coding(
        "k3", context_window=1_048_576, max_tokens=131_072
    ),
    "kimi-for-coding": _kimi_coding("kimi-for-coding"),
    "kimi-for-coding-highspeed": _kimi_coding("kimi-for-coding-highspeed"),
}


def resolve_model(model_id: str) -> ModelSpec:
    """Look up a model; unknown ids get a generic OpenAI-compatible spec."""
    return CATALOG.get(model_id) or ModelSpec(id=model_id, provider=GENERIC_OPENAI)


def resolve_base_url(spec: ModelSpec, configured: str) -> str:
    """Resolve the effective base URL for a model.

    An explicitly configured ``base_url`` always wins. When it is left at the
    global DeepSeek default but the catalog knows a different endpoint for the
    model's provider (e.g. switching to ``kimi-k3`` without editing
    ``base_url``), the provider endpoint is used so model switching works out
    of the box.
    """
    if configured != DEFAULT_BASE_URL or not spec.provider.base_url:
        return configured
    return spec.provider.base_url


def resolve_api_key(spec: ModelSpec, configured: str | None) -> str | None:
    """Resolve the API key: config first, then the provider's env var."""
    if configured:
        return configured
    if spec.provider.api_key_env:
        return os.environ.get(spec.provider.api_key_env)
    return None
