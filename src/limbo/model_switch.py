"""Runtime model switching (/model).

Domain logic for switching models mid-session, lifted out of the UI:
hot-reloading LLM config, validating the target model (API key, thinking
effort compatibility), swapping the client (converging on the latest
config model), and persisting the choice. The screen translates the
returned notices into chat messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from limbo.config import Config, load_config, save_model_to_config
from limbo.llm.catalog import (
    GENERIC_OPENAI,
    resolve_api_key,
    resolve_api_key_env,
    resolve_model,
)
from limbo.llm.client import LLMClient
from limbo.llm.factory import create_llm_client

if TYPE_CHECKING:
    from limbo.agent import Agent


@dataclass(frozen=True)
class ModelSwitchVerdict:
    """Outcome of validating a requested switch.

    ``notices`` are chat-ready info lines (already localized); the screen
    renders them verbatim.
    """

    switched: bool
    notices: list[str] = field(default_factory=list)


def reload_llm_config(config: Config) -> None:
    """Hot-reload llm/providers settings from config.toml.

    Only the LLM-relevant fields are replaced in place (the agent holds
    a reference to this Config), so edits made while Limbo is running —
    e.g. a newly added [providers.<id>] key — apply without a restart.
    """
    fresh = load_config()
    config.llm = fresh.llm
    config.providers = fresh.providers


def prepare_model_switch(model_id: str, config: Config) -> ModelSwitchVerdict:
    """Validate a /model switch and normalize config for the new model.

    On success ``config.llm.model`` is set (last, so a follow-up
    :func:`swap_llm_client` converges on it) and any incompatible
    ``thinking_effort`` is reset. No client changes happen here.
    """
    if model_id == config.llm.model:
        return ModelSwitchVerdict(False, [f"当前已是 {model_id}"])
    spec = resolve_model(model_id)
    if resolve_api_key(spec, config) is None:
        env = resolve_api_key_env(spec, config)
        hint = f"（${env}）" if env else ""
        return ModelSwitchVerdict(
            False,
            [f"未配置 {spec.provider.id} 的 API key{hint}，无法切换到 {model_id}"],
        )
    notices: list[str] = []
    effort = config.llm.thinking_effort
    if effort and spec.thinking_levels and effort not in spec.thinking_levels:
        config.llm.thinking_effort = None
        notices.append("当前 thinking_effort 不受新模型支持，已重置")
    if spec.provider is GENERIC_OPENAI:
        notices.append("未知模型，按 OpenAI 兼容默认参数接入")
    # Set the model on config last — the swap re-resolves everything from
    # config.llm.model, so rapid switches converge on the latest value.
    config.llm.model = model_id
    return ModelSwitchVerdict(True, notices)


async def swap_llm_client(
    config: Config, current_client: LLMClient, agent: Agent
) -> tuple[LLMClient, list[str]]:
    """Swap the client for the *current* config model.

    Reads config.llm.model at run time rather than a value captured when
    the command fired, so two rapid /model commands converge: the last
    swap leaves the runtime client and the config file pointing at the
    same (latest) model. Returns the new client and chat-ready notices.
    """
    model_id = config.llm.model
    close = getattr(current_client, "close", None)
    if close is not None:
        await close()
    new_client = create_llm_client(config)
    agent.update_llm(new_client)
    spec = resolve_model(model_id)
    notices = [f"已切换模型 {model_id} ({spec.provider.id})"]
    if not save_model_to_config(model_id):
        notices.append("配置写回失败，本次切换仅当前会话生效")
    return new_client, notices
