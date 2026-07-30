"""Tests for the startup ASCII-art banner on the main screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import Message
from limbo.sessions import SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.banner import DARK, RED, STARTUP_ART, YELLOW, startup_art_text
from limbo.ui.widgets.chat import ChatWidget


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover - make this an async generator


def make_app(tmp_path: Path, config: Config | None = None, **kwargs) -> LimboApp:
    cfg = config or Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_startup_art_shown_on_fresh_start(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        transcript = chat.transcript_text()
        # A distinctive mid-art line proves the banner was rendered verbatim.
        assert STARTUP_ART.splitlines()[len(STARTUP_ART.splitlines()) // 2] in transcript


@pytest.mark.asyncio
async def test_startup_art_hidden_when_resuming(tmp_path):
    session_path = tmp_path / "sessions" / "s1.jsonl"
    save_session(
        session_path,
        SessionMeta(id="s1", workdir=str(tmp_path.resolve()), title="old"),
        [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ],
    )
    app = make_app(tmp_path, resume=session_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        transcript = chat.transcript_text()
        assert STARTUP_ART.splitlines()[0] not in transcript
        assert "已恢复会话" in transcript


@pytest.mark.asyncio
async def test_startup_art_hidden_when_show_banner_false(tmp_path):
    """ui.show_banner = false skips the banner on fresh sessions (P0-5)."""
    cfg = Config()
    cfg.ui.show_banner = False
    app = make_app(tmp_path, config=cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        transcript = chat.transcript_text()
        assert STARTUP_ART.splitlines()[0] not in transcript
        assert "Limbo ready" in transcript


def test_startup_art_is_pure_ascii():
    assert STARTUP_ART.isascii()
    assert set(STARTUP_ART) <= {"@", "o", ".", " ", "\n"}


def test_startup_art_is_compact():
    """P0-5: the banner must not dominate the first screen (<= 8 rows)."""
    assert len(STARTUP_ART.splitlines()) <= 8


def test_startup_art_text_uses_palette_foreground_only():
    """Characters carry their palette color; spaces stay unstyled so the art
    blends into the theme background instead of painting a blue rectangle."""
    text = startup_art_text()
    lines = text.plain.splitlines()
    assert lines == STARTUP_ART.splitlines()

    def rgb(c: tuple[int, int, int]) -> str:
        return f"rgb({c[0]},{c[1]},{c[2]})"

    expected_fg = {"@": rgb(DARK), "o": rgb(RED), ".": rgb(YELLOW)}
    spans = list(text.spans)
    assert spans, "expected styled spans"
    covered = text.plain
    for span in spans:
        segment = covered[span.start : span.end]
        style = str(span.style)
        assert "on rgb" not in style  # no hardcoded background
        for ch in set(segment) - {"\n", " "}:
            assert expected_fg[ch] in style
