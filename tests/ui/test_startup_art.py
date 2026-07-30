"""Tests for the startup ASCII-art banner on the main screen."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import Message
from limbo.sessions import SessionMeta, save_session
from limbo.ui.app import LimboApp
from limbo.ui.banner import (
    BLUE,
    DARK,
    FULL,
    LOWER,
    RED,
    STARTUP_ART,
    UPPER,
    YELLOW,
    startup_art_text,
)
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
        art_lines = startup_art_text().plain.splitlines()
        assert art_lines[len(art_lines) // 2] in transcript


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
        assert UPPER not in transcript  # half-block banner not rendered
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
        assert UPPER not in transcript  # half-block banner not rendered
        assert "Limbo ready" in transcript


def test_startup_art_pixel_map_charset():
    """The pixel map stays pure ASCII: palette keys + transparent space."""
    assert STARTUP_ART.isascii()
    assert set(STARTUP_ART) <= {"R", "Y", "D", " ", "\n"}


def test_startup_art_is_compact():
    """P0-5: the banner must not dominate the first screen (<= 20 rows).

    Half-block rendering packs two pixel rows per terminal row, so the
    40-row pixel map occupies 20 terminal rows.
    """
    assert len(startup_art_text().plain.splitlines()) <= 20


def test_startup_art_text_uses_palette_colors_only():
    """Colored half-block cells carry palette colors; fully transparent
    cells stay unstyled so the art blends into the theme background
    instead of painting a blue rectangle."""
    text = startup_art_text()

    def rgb(c: tuple[int, int, int]) -> str:
        return f"rgb({c[0]},{c[1]},{c[2]})"

    palette_styles = {rgb(c) for c in (DARK, RED, YELLOW)}

    covered: set[int] = set()
    spans = list(text.spans)
    assert spans, "expected styled spans"
    for span in spans:
        segment = text.plain[span.start : span.end]
        assert set(segment) <= {UPPER, LOWER, FULL}
        style = str(span.style)
        # Every color in the style (fg, and bg for mixed pixel pairs)
        # comes from the palette — never the hardcoded background blue.
        assert rgb(BLUE) not in style
        colors = {part for part in style.split(" on ") if part.startswith("rgb(")}
        assert colors and colors <= palette_styles
        covered.update(range(span.start, span.end))

    # Coverage invariant: every half-block glyph is styled, every
    # space/newline is unstyled (transparent).
    for i, ch in enumerate(text.plain):
        if ch in {UPPER, LOWER, FULL}:
            assert i in covered
        else:
            assert i not in covered
