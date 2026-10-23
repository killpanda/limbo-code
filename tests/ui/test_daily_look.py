"""Daily mode look (Limbo Dawn): theme variant, badges, divider, sidebar."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.theme import BUILTIN_THEMES as TEXTUAL_THEMES

from limbo.config import Config
from limbo.ui import motion
from limbo.ui.app import LimboApp
from limbo.ui.contrast import TEXT_MIN, check_palettes, contrast_ratio
from limbo.ui.daily_look import base_name, daily_variant, variant_name
from limbo.ui.theme import LIMBO_DUSK, LIMBO_LIGHT
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.key_hints import DAILY_IDLE_HINTS, IDLE_HINTS, KeyHints
from limbo.ui.widgets.mode_divider import ModeDivider, divider_line
from limbo.ui.widgets.sidebar import Sidebar, memory_writes, page_label


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def make_app(tmp_path: Path, daily: bool = False, look: str = "dawn") -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    cfg.daily.look = look
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
        daily=daily,
    )


# -- theme variant ------------------------------------------------------------


def test_variant_names_round_trip():
    assert variant_name("limbo-dusk") == "limbo-dusk+daily"
    assert base_name("limbo-dusk+daily") == "limbo-dusk"
    assert base_name("nord") == "nord"


def test_dawn_variant_warms_limbo_themes_and_keeps_semantics():
    dawn = daily_variant(LIMBO_DUSK)
    assert dawn.name == "limbo-dusk+daily"
    assert dawn.primary != LIMBO_DUSK.primary
    assert dawn.variables["grad-from"] != LIMBO_DUSK.variables["grad-from"]
    # Brand yellow and semantic colours stay.
    assert dawn.accent == LIMBO_DUSK.accent
    assert dawn.error == LIMBO_DUSK.error
    assert dawn.variables["diff-added"] == LIMBO_DUSK.variables["diff-added"]
    assert daily_variant(LIMBO_LIGHT).dark is False


def test_dawn_palettes_pass_contrast_check():
    results, failures = check_palettes()
    assert failures == 0
    assert any(r.theme == "limbo-dusk+daily" for r in results)


@pytest.mark.parametrize("theme", list(TEXTUAL_THEMES.values()), ids=lambda t: t.name)
def test_derived_variant_keeps_text_readable(theme):
    dawn = daily_variant(theme)
    assert dawn.name == theme.name + "+daily"
    v = dawn.variables
    if theme.ansi:
        # Terminal colour names cannot be mixed: colours stay, badges only.
        assert dawn.primary == theme.primary
        return
    # Readable, or pushed all the way to the theme's own foreground.
    assert contrast_ratio(v["user-message-fg"], v["user-message-bg"]) >= TEXT_MIN or (
        v["user-message-fg"].lower() == str(dawn.foreground).lower()
    )
    assert v["grad-mid"] == str(theme.warning)


# -- motion / divider ---------------------------------------------------------


def test_divider_line_fills_width():
    line = divider_line("☀ Daily", 40)
    assert line.startswith("╱" * 6 + "  ☀ Daily  ")
    assert line.endswith("╱")


def test_sweep_reverse_lights_the_right_end_first():
    stops = ("#000000", "#000000")
    lines = ("x" * 30,)
    forward = motion.sweep_lines(lines, stops, "#000000", 6, total=24)
    backward = motion.sweep_lines(lines, stops, "#000000", 6, total=24, reverse=True)

    def lit(text, index):
        return text.spans[index].style.color.triplet != (0, 0, 0)

    assert lit(forward, 2) and not lit(forward, 27)
    assert lit(backward, 27) and not lit(backward, 2)


# -- sidebar helpers ------------------------------------------------------------


def test_page_label():
    assert page_label("https://www.example.com/help/returns/?x=1") == (
        "example.com",
        "/help/returns",
    )
    assert page_label("ftp://x") is None
    assert page_label("not a url") is None


def test_memory_writes_skips_removals():
    args = {
        "target": "user",
        "operations": [
            {"action": "add", "content": "更在意退货运费"},
            {"action": "remove", "old_text": "x"},
            {"action": "patch", "old_text": "a", "content": " b "},
        ],
    }
    assert memory_writes(args) == ["更在意退货运费", "b"]
    assert memory_writes({"operations": "nope"}) == []


# -- TUI wiring -----------------------------------------------------------------


def _dividers(screen) -> list[ModeDivider]:
    return list(screen.query_one("#chat", ChatWidget).query(ModeDivider))


@pytest.mark.asyncio
async def test_toggle_switches_theme_badges_hints_and_adds_divider(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert pilot.app.theme == "limbo-dusk"
        assert not _dividers(screen)

        screen._handle_command("/daily-mode on")
        await pilot.pause()
        assert pilot.app.theme == "limbo-dusk+daily"
        assert pilot.app.base_theme == "limbo-dusk"
        assert screen.has_class("-daily")
        assert screen.query_one("#hint", KeyHints)._hints == DAILY_IDLE_HINTS
        dividers = _dividers(screen)
        assert len(dividers) == 1 and dividers[0].daily
        assert "Daily mode" in str(dividers[0].render())
        sidebar = screen.query_one("#sidebar", Sidebar)
        assert sidebar.query_one("#sb-pages").display
        assert not sidebar.query_one("#sb-files").display

        screen._handle_command("/daily-mode on")  # unchanged: no new divider
        await pilot.pause()
        assert len(_dividers(screen)) == 1

        screen._handle_command("/daily-mode off")
        await pilot.pause()
        assert pilot.app.theme == "limbo-dusk"
        assert not screen.has_class("-daily")
        assert screen.query_one("#hint", KeyHints)._hints == IDLE_HINTS
        dividers = _dividers(screen)
        assert len(dividers) == 2 and not dividers[1].daily
        assert not sidebar.query_one("#sb-pages").display
        assert sidebar.query_one("#sb-files").display


@pytest.mark.asyncio
async def test_badge_look_keeps_colours(tmp_path):
    app = make_app(tmp_path, daily=True, look="badge")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert screen.agent.daily_mode
        assert pilot.app.theme == "limbo-dusk"
        assert screen.has_class("-daily")


@pytest.mark.asyncio
async def test_sidebar_records_pages_and_memory(tmp_path):
    app = make_app(tmp_path, daily=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        sidebar = pilot.app.screen.query_one("#sidebar", Sidebar)
        sidebar.record_tool_result("web_fetch", {"url": "https://www.a.com/x"})
        sidebar.record_tool_result("web_fetch", {"url": "https://b.org/"})
        sidebar.record_tool_result(
            "memory", {"target": "user", "operations": [{"action": "add", "content": "喜欢简洁"}]}
        )
        await pilot.pause()
        pages = str(sidebar.query_one("#sb-pages").render())
        assert pages.index("b.org") < pages.index("a.com")  # most recent first
        assert "喜欢简洁" in str(sidebar.query_one("#sb-memory").render())


@pytest.mark.asyncio
async def test_daily_landing_uses_daily_copy(tmp_path):
    app = make_app(tmp_path, daily=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        text = pilot.app.screen.query_one("#chat", ChatWidget).transcript_text()
        assert "白天的事" in text
