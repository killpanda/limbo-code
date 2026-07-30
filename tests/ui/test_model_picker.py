"""Tests for /model: model picker, runtime switching, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Label

import limbo.config as config_module
from limbo.config import Config, ProviderOverride
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.screens.model_picker import ModelPicker
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget
from limbo.ui.widgets.status_bar import StatusBar

KEY_ENVS = [
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "ZHIPUAI_API_KEY",
    "CODEX_API_KEY",
]

DEFAULT_FILE = '[llm]\napi_key = "test"\nmodel = "deepseek-chat"\n'


class FakeLLMClient:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = 0

    async def chat(self, messages, tools, on_request=None):
        self.calls += 1
        if self.responses:
            for event in self.responses.pop(0):
                yield event


@pytest.fixture(autouse=True)
def isolated_config_path(tmp_path, monkeypatch):
    """Point config.toml at a per-test file so /model hot-reload never
    touches the developer's real ~/.limbo/config.toml."""
    path = tmp_path / "config.toml"
    path.write_text(DEFAULT_FILE)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", path)
    return path


@pytest.fixture
def clear_key_envs(monkeypatch):
    for env in KEY_ENVS:
        monkeypatch.delenv(env, raising=False)


def make_app(tmp_path: Path, fake_llm=None, configure=None) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    if configure:
        configure(cfg)
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm or FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


def item_labels(picker) -> list[str]:
    return [
        str(item.query_one(Label).render())
        for item in picker.query("ListItem")
    ]


async def submit(pilot, text: str) -> None:
    screen = pilot.app.screen_stack[-1]
    input_widget = screen.query_one("#input", InputWidget)
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


async def wait_for_chat(pilot, needle: str) -> None:
    chat = pilot.app.screen.query_one("#chat", ChatWidget)
    for _ in range(200):
        await pilot.pause()
        if needle in chat.transcript_text():
            return
    raise AssertionError(f"timed out waiting for {needle!r} in chat")


# -- picker -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_command_opens_picker(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await pilot.pause()
        assert isinstance(pilot.app.screen_stack[-1], ModelPicker)


@pytest.mark.asyncio
async def test_picker_groups_by_provider_and_marks_current(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await pilot.pause()
        picker = pilot.app.screen_stack[-1]
        labels = item_labels(picker)
        headers = [t for t in labels if t.startswith("▸")]
        assert headers == [
            "▸ deepseek",
            "▸ moonshotai",
            "▸ kimi-coding",
            "▸ glm",
            "▸ codex",
        ]
        current = [t for t in labels if t.endswith("当前")]
        assert len(current) == 1
        assert current[0].strip().startswith("deepseek-chat")
        assert any(
            "glm-4.7" in t and "200K" in t and "reasoning" in t for t in labels
        )


@pytest.mark.asyncio
async def test_picker_marks_unavailable_providers(
    tmp_path, isolated_config_path, clear_key_envs, monkeypatch
):
    # No key in the file; only glm resolves one (from the env).
    isolated_config_path.write_text('[llm]\nmodel = "deepseek-chat"\n')
    monkeypatch.setenv("ZHIPUAI_API_KEY", "glm-key")

    def configure(cfg):
        cfg.llm.api_key = None

    app = make_app(tmp_path, configure=configure)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await pilot.pause()
        picker = pilot.app.screen_stack[-1]
        headers = [
            str(item.query_one(Label).render())
            for item in picker.query("ListItem.provider-header")
        ]
        assert headers[0] == "▸ deepseek（未配置 API key: $DEEPSEEK_API_KEY）"
        assert headers[3] == "▸ glm"  # only glm has a key
        assert len(picker.query("ListItem.unavailable")) > 0


@pytest.mark.asyncio
async def test_picker_select_unavailable_keeps_open_with_hint(
    tmp_path, isolated_config_path, clear_key_envs
):
    isolated_config_path.write_text('[llm]\nmodel = "deepseek-chat"\n')

    def configure(cfg):
        cfg.llm.api_key = None

    app = make_app(tmp_path, configure=configure)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await pilot.pause()
        picker = pilot.app.screen_stack[-1]
        list_view = picker.query_one("ListView")
        # Index 1 = first model row (deepseek-chat), unavailable without a key.
        list_view.index = 1
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.screen_stack[-1] is picker
        hint = str(picker.query_one("#picker-hint").render())
        assert "未配置 deepseek 的 API key" in hint


@pytest.mark.asyncio
async def test_picker_escape_dismisses_without_switch(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(pilot.app.screen_stack[-1], MainScreen)
        assert pilot.app.screen.config.llm.model == "deepseek-chat"


@pytest.mark.asyncio
async def test_picker_select_switches_model(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        await pilot.pause()
        picker = pilot.app.screen_stack[-1]
        labels = item_labels(picker)
        target = next(i for i, t in enumerate(labels) if "glm-4.7" in t)
        picker.query_one("ListView").index = target
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(pilot.app.screen_stack[-1], MainScreen)
        await wait_for_chat(pilot, "已切换模型 glm-4.7 (glm)")


# -- switching ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_switch_updates_everything(tmp_path):
    fake = FakeLLMClient()
    app = make_app(tmp_path, fake_llm=fake)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        old_client = screen.llm_client
        await submit(pilot, "/model glm-4.7")
        await wait_for_chat(pilot, "已切换模型 glm-4.7 (glm)")

        assert screen.config.llm.model == "glm-4.7"
        assert screen.llm_client is not old_client
        assert isinstance(screen.llm_client, OpenAICompatibleClient)
        assert screen.agent.llm_client is screen.llm_client
        assert screen.agent._context_window == 204_800
        statusbar = screen.query_one("#statusbar", StatusBar)
        assert statusbar._model == "glm-4.7"


@pytest.mark.asyncio
async def test_switch_to_same_model_is_noop(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        old_client = screen.llm_client
        await submit(pilot, "/model deepseek-chat")
        await wait_for_chat(pilot, "当前已是 deepseek-chat")
        assert screen.llm_client is old_client


@pytest.mark.asyncio
async def test_switch_without_key_refused(
    tmp_path, isolated_config_path, clear_key_envs
):
    isolated_config_path.write_text('[llm]\nmodel = "deepseek-chat"\n')

    def configure(cfg):
        cfg.llm.api_key = None

    app = make_app(tmp_path, configure=configure)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        old_client = screen.llm_client
        await submit(pilot, "/model glm-4.7")
        await wait_for_chat(pilot, "未配置 glm 的 API key（$ZHIPUAI_API_KEY）")
        assert screen.config.llm.model == "deepseek-chat"
        assert screen.llm_client is old_client


@pytest.mark.asyncio
async def test_switch_unknown_model_uses_generic_fallback(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        await submit(pilot, "/model my-local-model")
        await wait_for_chat(pilot, "已切换模型 my-local-model (custom)")
        await wait_for_chat(pilot, "未知模型，按 OpenAI 兼容默认参数接入")
        assert screen.agent._context_window == 128_000


@pytest.mark.asyncio
async def test_rapid_switches_converge_to_last_model(
    tmp_path, isolated_config_path
):
    """Two rapid /model commands must not leave client/status bar/config
    pointing at different models (review nit: the swap worker reads the
    live config.llm.model, not the value captured at command time)."""
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        await submit(pilot, "/model glm-4.7")
        await submit(pilot, "/model glm-5.2")
        await wait_for_chat(pilot, "已切换模型 glm-5.2 (glm)")
        for _ in range(50):
            await pilot.pause()

        assert screen.config.llm.model == "glm-5.2"
        assert screen.llm_client.spec.id == "glm-5.2"
        assert screen.agent._context_window == 1_000_000
        statusbar = screen.query_one("#statusbar", StatusBar)
        assert statusbar._model == "glm-5.2"
        assert 'model = "glm-5.2"' in isolated_config_path.read_text()


@pytest.mark.asyncio
async def test_busy_guard_blocks_switch(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        screen._agent_busy = True
        await submit(pilot, "/model glm-4.7")
        await wait_for_chat(pilot, "当前任务进行中")
        assert pilot.app.screen_stack[-1] is screen
        assert screen.config.llm.model == "deepseek-chat"


@pytest.mark.asyncio
async def test_thinking_effort_reset_when_unsupported(
    tmp_path, isolated_config_path
):
    isolated_config_path.write_text(
        '[llm]\napi_key = "test"\nmodel = "deepseek-chat"\n'
        'thinking_effort = "max"\n\n'  # gpt-5.5 tops out at xhigh
        '[providers.codex]\napi_key = "codex-key"\n'
    )

    def configure(cfg):
        cfg.llm.thinking_effort = "max"
        cfg.providers["codex"] = ProviderOverride(api_key="codex-key")

    app = make_app(tmp_path, configure=configure)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        await submit(pilot, "/model gpt-5.5")
        await wait_for_chat(pilot, "已切换模型 gpt-5.5 (codex)")
        await wait_for_chat(pilot, "thinking_effort 不受新模型支持，已重置")
        assert screen.config.llm.thinking_effort is None


# -- hot reload & persistence ---------------------------------------------------


@pytest.mark.asyncio
async def test_hot_reload_picks_up_new_provider_key(
    tmp_path, isolated_config_path, clear_key_envs
):
    isolated_config_path.write_text('[llm]\nmodel = "deepseek-chat"\n')

    def configure(cfg):
        cfg.llm.api_key = None

    app = make_app(tmp_path, configure=configure)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        from limbo.llm.catalog import resolve_api_key, resolve_model

        screen._reload_llm_config()
        assert resolve_api_key(resolve_model("glm-4.7"), screen.config) is None

        isolated_config_path.write_text(
            '[llm]\nmodel = "deepseek-chat"\n\n'
            '[providers.glm]\napi_key = "fresh-glm-key"\n'
        )
        await submit(pilot, "/model glm-4.7")
        await wait_for_chat(pilot, "已切换模型 glm-4.7 (glm)")


@pytest.mark.asyncio
async def test_switch_persists_model_preserving_comments(
    tmp_path, isolated_config_path
):
    isolated_config_path.write_text(
        '# my limbo config\n[llm]\napi_key = "test"\n'
        'model = "deepseek-chat"  # default\n'
    )

    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(pilot, "/model glm-4.7")
        await wait_for_chat(pilot, "已切换模型 glm-4.7 (glm)")

    text = isolated_config_path.read_text()
    assert 'model = "glm-4.7"' in text
    assert "# my limbo config" in text  # tomlkit round-trip keeps comments
    assert 'api_key = "test"' in text


@pytest.mark.asyncio
async def test_switch_degrades_when_write_back_fails(
    tmp_path, isolated_config_path, clear_key_envs, monkeypatch
):
    monkeypatch.setenv("ZHIPUAI_API_KEY", "glm-key")
    isolated_config_path.write_text("[unclosed = ")  # malformed: never clobbered

    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        screen = pilot.app.screen_stack[-1]
        await submit(pilot, "/model glm-4.7")
        await wait_for_chat(pilot, "配置写回失败，本次切换仅当前会话生效")
        assert screen.config.llm.model == "glm-4.7"  # session switch worked

    assert isolated_config_path.read_text() == "[unclosed = "
