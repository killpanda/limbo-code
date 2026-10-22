"""Following the Omarchy system theme (Limbo Dusk design, step 6)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import limbo.ui.omarchy as omarchy
from limbo.config import Config
from limbo.ui.app import LimboApp

FIXTURES = Path(__file__).parent.parent / "fixtures" / "omarchy"
def install(state: Path, name: str) -> None:
    """Lay out an Omarchy state dir the way ``omarchy theme set`` does."""
    (state / "theme").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / f"{name}.toml", state / "theme" / "colors.toml")
    (state / "theme.name").write_text(name)


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def make_app(tmp_path: Path, theme: str | None = None) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.ui.theme = theme
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


@pytest.mark.asyncio
async def test_auto_follows_omarchy(tmp_path):
    install(omarchy.state_dir(), "tokyo-night")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "omarchy"
        assert pilot.app.current_theme.background == "#1A1B26"


