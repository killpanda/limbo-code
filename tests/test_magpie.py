"""Magpie local gateway provider: id mapping, defaults, wire model, discovery."""

from __future__ import annotations

import io
import json

import httpx
import pytest
import respx
from httpx import Response

from limbo import auth, cli_auth
from limbo.config import Config
from limbo.llm.catalog import (
    api_key_source,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)
from limbo.llm.factory import create_llm_client
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.models import Message

MODELS_URL = "http://127.0.0.1:3425/v1/models"


def test_magpie_inherits_catalog_metadata():
    spec = resolve_model("magpie/deepseek/deepseek-v4-pro")
    assert spec.provider.id == "magpie"
    assert spec.wire_id == "deepseek/deepseek-v4-pro"
    assert spec.context_window == 1_000_000
    assert spec.reasoning
    # Thinking goes through reasoning_effort, which Magpie translates.
    assert spec.thinking_format == "openai"
    assert set(spec.thinking_levels) == {"low", "medium", "high"}


def test_magpie_keeps_model_specific_levels_and_replay():
    spec = resolve_model("magpie/moonshot/kimi-k3")
    assert set(spec.thinking_levels) == {"low", "high", "max"}
    assert spec.requires_reasoning_content


def test_magpie_unknown_and_group_ids_get_generic_defaults():
    for model_id in ("magpie/group/coder", "magpie/auto/claude-opus-4.6"):
        spec = resolve_model(model_id)
        assert spec.provider.id == "magpie"
        assert spec.wire_id == model_id.removeprefix("magpie/")
        assert not spec.reasoning


def test_non_magpie_ids_unchanged():
    assert resolve_model("deepseek-v4-pro").wire_id == "deepseek-v4-pro"
    assert resolve_model("openrouter/some-model").provider.id == "custom"


def test_magpie_works_without_any_key(monkeypatch):
    monkeypatch.delenv("MAGPIE_API_KEY", raising=False)
    spec = resolve_model("magpie/codex/gpt-5.5")
    config = Config()
    assert resolve_api_key(spec, config) == "magpie"
    assert api_key_source(spec, config) == "默认 key"
    assert resolve_base_url(spec, config) == "http://127.0.0.1:3425/v1"
    monkeypatch.setenv("MAGPIE_API_KEY", "tok")
    assert resolve_api_key(spec, config) == "tok"


@pytest.mark.asyncio
@respx.mock
async def test_client_sends_magpie_model_name():
    route = respx.post("http://127.0.0.1:3425/v1/chat/completions").mock(
        return_value=Response(
            200,
            text='data: {"id":"1","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
    )
    config = Config()
    config.llm.model = "magpie/deepseek/deepseek-v4-pro"
    config.llm.thinking_effort = "high"
    client = create_llm_client(config)
    assert isinstance(client, OpenAICompatibleClient)
    _ = [e async for e in client.chat([Message(role="user", content="hi")], tools=[])]
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "deepseek/deepseek-v4-pro"
    assert body["reasoning_effort"] == "high"
    assert "thinking" not in body
    assert route.calls.last.request.headers["authorization"] == "Bearer magpie"


@respx.mock
def test_limbo_models_lists_live_magpie_models():
    respx.get(MODELS_URL).mock(
        return_value=Response(200, json={"data": [{"id": "groq/llama-3.1-8b-instant"}]})
    )
    out = io.StringIO()
    cli_auth.list_models(Config(), "magpie", stdout=out)
    assert "magpie/groq/llama-3.1-8b-instant" in out.getvalue()
    assert "magpie/codex/gpt-5.5" not in out.getvalue()


@respx.mock
def test_limbo_models_falls_back_when_gateway_down():
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("refused"))
    out = io.StringIO()
    cli_auth.list_models(Config(), "magpie", stdout=out)
    assert "magpie/codex/gpt-5.5" in out.getvalue()


def test_login_empty_input_uses_default_key():
    rc = cli_auth.login(
        "magpie", verify=False, stdin=io.StringIO("\n"), stdout=io.StringIO()
    )
    assert rc == 0
    assert auth.get("magpie").key == "magpie"
