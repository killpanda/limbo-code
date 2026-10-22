"""Tests for /model: model picker, runtime switching, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Label

import limbo.config as config_module
from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.screens.auth_dialog import AuthDialog
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget

KEY_ENVS = [
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "ZHIPUAI_API_KEY",
    "CODEX_API_KEY",
]

DEFAULT_FILE = '[llm]\napi_key = "test"\nmodel = "deepseek-v4-pro"\n'


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


# -- hot reload & persistence ---------------------------------------------------


# -- Limbo Dusk model dialog: search, inline login, thinking -----------------


@pytest.fixture
def fake_key_check(monkeypatch):
    from limbo.llm import verify

    seen = []

    async def check(provider, key, base_url=None, timeout=8.0):
        seen.append((provider.id, key, base_url))
        if key == "bad":
            return verify.KeyCheck(verify.INVALID, "拒绝")
        return verify.KeyCheck(verify.VALID, "key 有效")

    monkeypatch.setattr(verify, "check_key", check)
    return seen


@pytest.mark.asyncio
async def test_inline_login_saves_key_and_switches(
    tmp_path, isolated_config_path, clear_key_envs, fake_key_check
):
    from limbo import auth

    isolated_config_path.write_text('[llm]\nmodel = "deepseek-v4-pro"\n')

    def configure(cfg):
        cfg.llm.api_key = None

    app = make_app(tmp_path, configure=configure)
    async with app.run_test() as pilot:
        await submit(pilot, "/model")
        picker = pilot.app.screen_stack[-1]
        await pilot.press(*"glm-4.7")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        dialog = pilot.app.screen_stack[-1]
        assert isinstance(dialog, AuthDialog)
        await pilot.press(*"bad")
        await pilot.press("enter")
        await pilot.pause()
        assert pilot.app.screen_stack[-1] is dialog  # rejected key keeps it open
        dialog.query_one("#auth-key").value = "glm-secret"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert picker not in pilot.app.screen_stack
        await wait_for_chat(pilot, "已切换模型 glm-4.7 (glm)")
        assert auth.get("glm").key == "glm-secret"


