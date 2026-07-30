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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limbo.config import Config

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_TOKENS = 16_384

# API dialects. Each has a client implementation registered in
# limbo.llm.factory; the factory raises a clear error for anything else.
API_OPENAI_COMPLETIONS = "openai-completions"
API_ANTHROPIC_MESSAGES = "anthropic-messages"
API_OPENAI_RESPONSES = "openai-responses"

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
    # Extra request-body fields sent on every request (provider quirks not
    # covered by the OpenAI SDK's typed parameters).
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Extra request-body fields sent only when the request carries tools
    # (GLM Coding Plan expects tool_stream for streamed tool calls).
    tool_extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Per-model API characteristics."""

    id: str
    provider: ProviderSpec
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_tokens: int = DEFAULT_MAX_TOKENS
    reasoning: bool = False
    # Whether the model accepts image input (multimodal). Conservative by
    # default: unknown models (GENERIC_OPENAI fallback) stay False and image
    # attachments degrade to path references instead of breaking the call.
    vision: bool = False
    # How thinking is controlled: "openai" (reasoning_effort parameter),
    # "deepseek" (thinking: {type: enabled|disabled}), "zai" (deepseek shape
    # plus clear_thinking: false, with optional reasoning_effort passthrough
    # for models that declare thinking_levels), "openai-responses"
    # (reasoning: {effort, summary} on the Responses API), or None (the
    # model reasons but the API exposes no switch, e.g. deepseek-reasoner).
    thinking_format: str | None = None
    # Supported thinking levels mapped to provider values (openai format;
    # also drives reasoning_effort passthrough for the zai format).
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

# GLM Coding Plan (subscription): OpenAI-compatible dialect on a dedicated
# coding endpoint. Model metadata mirrors pi's zai-coding-cn provider
# (providers/data/zai-coding-cn.json); coding-plan keys only work on the
# coding endpoint (not the pay-per-token /api/paas/v4 one) and vice versa.
# International endpoint: https://api.z.ai/api/coding/paas/v4 — select via a
# [providers.glm] base_url override.
GLM_CODING = ProviderSpec(
    id="glm",
    api=API_OPENAI_COMPLETIONS,
    base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    api_key_env="ZHIPUAI_API_KEY",
    # pi sets tool_stream on all zai-coding tool requests (zaiToolStream).
    tool_extra_body={"tool_stream": True},
)

# OpenAI Codex via the Responses API. Model metadata mirrors pi's
# openai-codex provider (providers/data/openai-codex.json) — note pi targets
# the ChatGPT subscription backend (OAuth); limbo targets API-key relays, so
# point [providers.codex] base_url at your relay and make sure the model IDs
# below match what the relay actually exposes.
CODEX = ProviderSpec(
    id="codex",
    api=API_OPENAI_RESPONSES,
    base_url="https://api.openai.com/v1",
    api_key_env="CODEX_API_KEY",
)


def _codex(
    model_id: str,
    *,
    context_window: int = 272_000,
    max_tokens: int = 128_000,
    vision: bool = True,
    supports_max: bool = False,
) -> ModelSpec:
    # Codex models reason with Responses reasoning effort. pi's
    # thinkingLevelMap: standard levels through "high" pass through by the
    # provider default mapping, "minimal" maps to "low", "xhigh" is
    # explicit, and 5.6-generation models add "max".
    levels = {"low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh"}
    if supports_max:
        levels["max"] = "max"
    return ModelSpec(
        id=model_id,
        provider=CODEX,
        context_window=context_window,
        max_tokens=max_tokens,
        reasoning=True,
        vision=vision,
        thinking_format="openai-responses",
        thinking_levels=levels,
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
    vision: bool = False,
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
        vision=vision,
        thinking_format=thinking_format,
        thinking_levels=thinking_levels or {},
        thinking_can_disable=thinking_can_disable,
        requires_reasoning_content=requires_reasoning_content,
    )


def _glm(
    model_id: str,
    *,
    context_window: int,
    max_tokens: int = 131_072,
    vision: bool = False,
    thinking_levels: dict[str, str] | None = None,
) -> ModelSpec:
    # All GLM Coding Plan models reason with the zai thinking dialect
    # (thinking: {type: enabled, clear_thinking: false}); glm-5.2 also
    # accepts reasoning_effort (declared via thinking_levels).
    return ModelSpec(
        id=model_id,
        provider=GLM_CODING,
        context_window=context_window,
        max_tokens=max_tokens,
        reasoning=True,
        vision=vision,
        thinking_format="zai",
        thinking_levels=thinking_levels or {},
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
    # Kimi K2.5 is Moonshot's multimodal generation (accepts image input).
    "kimi-k2.5": _moonshot(
        "kimi-k2.5", reasoning=True, thinking_format="deepseek", vision=True
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
    # -- GLM Coding Plan (subscription whitelist) ----------------------------
    "glm-4.5-air": _glm("glm-4.5-air", context_window=131_072, max_tokens=98_304),
    "glm-4.7": _glm("glm-4.7", context_window=204_800),
    "glm-5-turbo": _glm("glm-5-turbo", context_window=200_000),
    "glm-5.1": _glm("glm-5.1", context_window=200_000),
    # glm-5.2: 1M context; reasoning_effort supported — pi's thinkingLevelMap
    # maps low/medium→high (minimal unsupported).
    "glm-5.2": _glm(
        "glm-5.2",
        context_window=1_000_000,
        thinking_levels={"low": "high", "high": "high", "max": "max"},
    ),
    "glm-5v-turbo": _glm("glm-5v-turbo", context_window=200_000, vision=True),
    # -- OpenAI Codex (Responses API dialect) --------------------------------
    "gpt-5.3-codex-spark": _codex(
        "gpt-5.3-codex-spark", context_window=128_000, vision=False
    ),
    "gpt-5.4": _codex("gpt-5.4"),
    "gpt-5.4-mini": _codex("gpt-5.4-mini"),
    "gpt-5.5": _codex("gpt-5.5"),
    "gpt-5.6-sol": _codex("gpt-5.6-sol", supports_max=True),
    "gpt-5.6-terra": _codex("gpt-5.6-terra", supports_max=True),
    "gpt-5.6-luna": _codex("gpt-5.6-luna", supports_max=True),
}


def resolve_model(model_id: str) -> ModelSpec:
    """Look up a model; unknown ids get a generic OpenAI-compatible spec."""
    return CATALOG.get(model_id) or ModelSpec(id=model_id, provider=GENERIC_OPENAI)


def resolve_base_url(spec: ModelSpec, config: Config) -> str:
    """Resolve the effective base URL for a model.

    Resolution order (first hit wins):

    1. ``[providers.<id>] base_url`` — per-provider override. This is the
       one exception to "explicit [llm] base_url always wins": a provider
       override is more specific than the global setting.
    2. ``[llm] base_url`` when explicitly changed from the DeepSeek default.
    3. The catalog provider's built-in endpoint, so switching to a catalog
       model (e.g. ``kimi-k3``, ``glm-4.7``) works without editing base_url.
    4. The configured value (DeepSeek default) as the final fallback.
    """
    override = config.providers.get(spec.provider.id)
    if override and override.base_url:
        return override.base_url
    configured = config.llm.base_url
    if configured != DEFAULT_BASE_URL or not spec.provider.base_url:
        return configured
    return spec.provider.base_url


def resolve_api_key(spec: ModelSpec, config: Config) -> str | None:
    """Resolve the API key for a model.

    Resolution order (first hit wins): ``[providers.<id>] api_key``, then
    ``[llm] api_key``, then the environment variable named by
    ``resolve_api_key_env``.
    """
    override = config.providers.get(spec.provider.id)
    if override and override.api_key:
        return override.api_key
    if config.llm.api_key:
        return config.llm.api_key
    env = resolve_api_key_env(spec, config)
    if env:
        return os.environ.get(env)
    return None


def resolve_api_key_env(spec: ModelSpec, config: Config) -> str | None:
    """Effective credential env var name (``[providers.<id>] api_key_env`` wins)."""
    override = config.providers.get(spec.provider.id)
    if override and override.api_key_env:
        return override.api_key_env
    return spec.provider.api_key_env


def resolve_headers(spec: ModelSpec, config: Config) -> dict[str, str]:
    """Provider headers merged with ``[providers.<id>] headers`` (override wins)."""
    override = config.providers.get(spec.provider.id)
    if override:
        return {**spec.provider.headers, **override.headers}
    return dict(spec.provider.headers)
