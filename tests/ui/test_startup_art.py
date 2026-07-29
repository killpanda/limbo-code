"""Tests for the startup ASCII-art banner on the main screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import Message
from limbo.sessions import SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.banner import BLUE, DARK, RED, STARTUP_ART, YELLOW, startup_art_text
from limbo.ui.widgets.chat import ChatWidget


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover - make this an async generator


def make_app(tmp_path: Path, **kwargs) -> LimboApp:
    cfg = Config()
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


def test_startup_art_is_pure_ascii():
    assert STARTUP_ART.isascii()
    assert set(STARTUP_ART) <= {"@", "o", ".", " ", "\n"}


def test_startup_art_text_uses_original_colors():
    text = startup_art_text()
    # Plain text is the art itself, padded to a uniform rectangle.
    lines = text.plain.splitlines()
    assert len(lines) == len(STARTUP_ART.splitlines())
    assert len({len(line) for line in lines}) == 1
    for raw, padded in zip(STARTUP_ART.splitlines(), lines, strict=True):
        assert padded.startswith(raw)

    # Every character carries the palette color of its fill on the blue bg.
    def rgb(c: tuple[int, int, int]) -> str:
        return f"rgb({c[0]},{c[1]},{c[2]})"

    expected_fg = {"@": rgb(DARK), "o": rgb(RED), ".": rgb(YELLOW), " ": rgb(BLUE)}
    bg = rgb(BLUE)
    spans = list(text.spans)
    assert spans, "expected styled spans"
    covered = text.plain
    for span in spans:
        segment = covered[span.start : span.end]
        style = str(span.style)
        for ch in set(segment) - {"\n"}:
            assert expected_fg[ch] in style
            assert f"on {bg}" in style
