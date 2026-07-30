import pytest

from limbo.ui.app import LimboApp


@pytest.mark.asyncio
async def test_app_mounts(tmp_path):
    app = LimboApp(workdir=".", session_dir=tmp_path / "sessions")
    async with app.run_test() as pilot:
        assert pilot.app.is_running


@pytest.mark.asyncio
async def test_main_screen_uses_factory_client_for_kimi_coding(tmp_path):
    """MainScreen must pick the client via the provider factory, not
    hardcode the OpenAI-compatible client (regression: kimi-coding models
    were sent OpenAI-format requests, yielding 404s)."""
    from limbo.config import Config
    from limbo.llm.anthropic_client import AnthropicMessagesClient
    from limbo.llm.openai_client import OpenAICompatibleClient
    from limbo.ui.screens.main import MainScreen

    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.llm.model = "k3"
    screen = MainScreen(workdir=tmp_path, config=cfg, session_dir=tmp_path / "sessions")
    assert isinstance(screen.llm_client, AnthropicMessagesClient)

    cfg.llm.model = "deepseek-chat"
    screen = MainScreen(workdir=tmp_path, config=cfg, session_dir=tmp_path / "sessions")
    assert isinstance(screen.llm_client, OpenAICompatibleClient)

    # Isolation guard: nothing the screen/agent writes may land outside
    # tmp_path (a missing session_dir used to leak trace files into the
    # developer's real ~/.limbo/sessions).
    assert screen.agent.trace.path.is_relative_to(tmp_path)
