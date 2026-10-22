"""/tracing-viewer in the TUI: registered, usable mid-turn, reports the URL."""

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget


class IdleLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield


@pytest.mark.asyncio
async def test_tracing_viewer_command_starts_server_and_reports_url(tmp_path):
    cfg = Config()
    cfg.llm.api_key = "test"
    app = LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=IdleLLMClient(),
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen_stack[-1]
        assert isinstance(screen, MainScreen)
        command = screen._commands.get("/tracing-viewer")
        assert command is not None
        assert command.allow_when_busy is True  # read-only: watch a turn live

        opened: list[str] = []
        viewer_command = screen._tracing_viewer
        viewer_command._browser_available = lambda: True
        viewer_command._open_browser = lambda url: opened.append(url) or True

        command.handler("")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert viewer_command.viewer is not None and viewer_command.viewer.running
        assert opened == [viewer_command.viewer.url(screen.agent.session_id)]
        chat = screen.query_one("#chat", ChatWidget)
        assert "Tracing viewer 已在浏览器打开" in chat.transcript_text()
        viewer = viewer_command.viewer
    # Unmount stops the server with the app.
    assert not viewer.running
