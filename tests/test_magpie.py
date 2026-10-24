"""Magpie local gateway provider: id mapping, live metadata, wire model, discovery."""

from __future__ import annotations

import asyncio
import io
import json

import httpx
import pytest
import respx
from httpx import Response

from limbo import auth, cli_auth
from limbo.config import Config, ProviderOverride
from limbo.llm import magpie
from limbo.llm.catalog import (
    API_ANTHROPIC_MESSAGES,
    API_OPENAI_COMPLETIONS,
    API_OPENAI_RESPONSES,
    MAGPIE_LIVE,
    api_key_source,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    thinking_choices,
)
from limbo.llm.factory import create_llm_client
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.models import Message

MODELS_URL = "http://127.0.0.1:3425/v1/models"

# Shapes as the real gateway returns them (trimmed).
GATEWAY_MODELS = [
    {
        "id": "deepseek/deepseek-flash",
        "context_length": 1_000_000,
        "context_window": 1_000_000,
        "max_output_tokens": 384_000,
        "modalities": {"input": ["text", "image"], "output": ["text"]},
        "reasoning": True,
        "reasoning_levels": ["none", "low", "high", "max"],
        "native_endpoints": ["/v1/chat/completions", "/v1/responses", "/v1/messages"],
    },
    {
        "id": "opencode-go/grok-4.7",
        "context_length": 500_000,
        "max_output_tokens": 32_000,
        "reasoning": True,
        "reasoning_levels": ["low", "high"],
        "native_endpoints": ["/v1/responses"],
    },
    {
        "id": "opencode-go/claude-haiku-5-5",
        "context_length": 200_000,
        "reasoning": False,
        "reasoning_levels": [],
        "native_endpoints": ["/v1/messages"],
    },
    {"id": "auto/deepseek-v4-pro", "context_length": None},
]


@pytest.fixture(autouse=True)
def clear_live():
    MAGPIE_LIVE.clear()
    yield
    MAGPIE_LIVE.clear()


def _load(models=GATEWAY_MODELS):
    respx.get(MODELS_URL).mock(return_value=Response(200, json={"data": models}))
    return asyncio.run(magpie.live_models(Config()))


def test_offline_ids_guess_from_catalog():
    spec = resolve_model("magpie/deepseek/deepseek-v4-pro")
    assert spec.provider.id == "magpie"
    assert spec.wire_id == "deepseek/deepseek-v4-pro"
    assert spec.context_window == 1_000_000
    assert spec.thinking_format == "openai"


def test_unknown_ids_get_generic_defaults():
    spec = resolve_model("magpie/group/coder")
    assert spec.provider.id == "magpie"
    assert spec.wire_id == "group/coder"
    assert not spec.reasoning


def test_non_magpie_ids_unchanged():
    assert resolve_model("deepseek-v4-pro").wire_id == "deepseek-v4-pro"
    assert resolve_model("openrouter/some-model").provider.id == "custom"


@respx.mock
def test_live_metadata_drives_spec():
    specs = _load()
    assert specs[0].id == "magpie/deepseek/deepseek-flash"
    flash = resolve_model("magpie/deepseek/deepseek-flash")
    assert flash.provider.api == API_OPENAI_COMPLETIONS
    assert flash.context_window == 1_000_000
    assert flash.max_tokens == 65_536  # catalog cap, below the gateway max
    assert flash.vision
    assert flash.thinking_levels == {"off": "none", "low": "low", "high": "high", "max": "max"}
    assert "off" in thinking_choices(flash)

    grok = resolve_model("magpie/opencode-go/grok-4.7")
    assert grok.provider.api == API_OPENAI_RESPONSES
    assert grok.context_window == 500_000
    assert grok.max_tokens == 32_000
    assert grok.thinking_format == "openai-responses"
    assert set(grok.thinking_levels) == {"low", "high"}

    haiku = resolve_model("magpie/opencode-go/claude-haiku-5-5")
    assert haiku.provider.api == API_ANTHROPIC_MESSAGES
    assert not haiku.reasoning
    assert resolve_base_url(haiku, Config()) == "http://127.0.0.1:3425"

    auto = resolve_model("magpie/auto/deepseek-v4-pro")
    assert auto.provider.api == API_OPENAI_COMPLETIONS
    assert auto.context_window == 1_000_000  # null length → catalog guess


def test_messages_variant_strips_v1_from_override():
    MAGPIE_LIVE["x/claude"] = {"id": "x/claude", "native_endpoints": ["/v1/messages"]}
    config = Config()
    config.providers["magpie"] = ProviderOverride(base_url="http://box:3425/v1")
    assert resolve_base_url(resolve_model("magpie/x/claude"), config) == "http://box:3425"


def test_magpie_works_without_any_key(monkeypatch):
    monkeypatch.delenv("MAGPIE_API_KEY", raising=False)
    spec = resolve_model("magpie/deepseek/deepseek-flash")
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
    respx.get(MODELS_URL).mock(return_value=Response(200, json={"data": GATEWAY_MODELS}))
    out = io.StringIO()
    cli_auth.list_models(Config(), "grok", stdout=out)
    text = out.getvalue()
    assert "magpie/opencode-go/grok-4.7" in text
    assert "488K" in text  # the gateway's 500_000 window
    assert "思考 low/high" in text


@respx.mock
def test_limbo_models_falls_back_when_gateway_down():
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("refused"))
    out = io.StringIO()
    cli_auth.list_models(Config(), "magpie", stdout=out)
    assert "magpie/deepseek/deepseek-flash" in out.getvalue()


@respx.mock
def test_startup_loads_live_list_once():
    route = respx.get(MODELS_URL).mock(
        return_value=Response(200, json={"data": GATEWAY_MODELS})
    )
    magpie.load_for_startup("magpie/opencode-go/grok-4.7", Config())
    magpie.load_for_startup("magpie/opencode-go/grok-4.7", Config())
    assert route.call_count == 1
    assert resolve_model("magpie/opencode-go/grok-4.7").context_window == 500_000
    magpie.load_for_startup("deepseek-v4-pro", Config())
    assert route.call_count == 1


def test_login_empty_input_uses_default_key():
    rc = cli_auth.login(
        "magpie", verify=False, stdin=io.StringIO("\n"), stdout=io.StringIO()
    )
    assert rc == 0
    assert auth.get("magpie").key == "magpie"


@pytest.mark.asyncio
@respx.mock
async def test_model_picker_reads_live_magpie_models(tmp_path):
    from textual.widgets import Label

    from limbo.ui.app import LimboApp
    from limbo.ui.screens.model_picker import ModelPicker

    respx.get(MODELS_URL).mock(
        return_value=Response(200, json={"data": [{"id": "groq/llama-3.1-8b-instant"}]})
    )
    config = Config()
    config.llm.api_key = "test"
    app = LimboApp(workdir=tmp_path, config=config, session_dir=tmp_path / "s")
    async with app.run_test() as pilot:
        picker = ModelPicker(config, config.llm.model)
        app.push_screen(picker)
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        labels = [str(item.query_one(Label).render()) for item in picker.query("ListItem")]
    assert any("magpie/groq/llama-3.1-8b-instant" in label for label in labels)
    assert not any("magpie/deepseek/deepseek-flash" in label for label in labels)
