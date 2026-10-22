"""Following the Omarchy system theme (Limbo Dusk design, step 6)."""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

import pytest

import limbo.ui.omarchy as omarchy
from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.contrast import TEXT_MIN, USED_PAIRS, check_theme, contrast_ratio
from limbo.ui.theme import DEFAULT_THEME, DUSK_VARIABLES, palette_snapshot

FIXTURES = Path(__file__).parent.parent / "fixtures" / "omarchy"
THEMES = sorted(p.stem for p in FIXTURES.glob("*.toml"))


def load_fixture(name: str) -> dict[str, object]:
    with (FIXTURES / f"{name}.toml").open("rb") as fh:
        return tomllib.load(fh)


def install(state: Path, name: str) -> None:
    """Lay out an Omarchy state dir the way ``omarchy theme set`` does."""
    (state / "theme").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / f"{name}.toml", state / "theme" / "colors.toml")
    (state / "theme.name").write_text(name)


@pytest.mark.parametrize("name", THEMES)
def test_every_used_text_pair_is_readable(name):
    theme = omarchy.build_theme(load_fixture(name))
    results = check_theme(theme.name, palette_snapshot(theme))
    failing = [
        f"{r.fg} on {r.bg} = {r.ratio:.2f}"
        for r in results
        if (r.fg, r.bg) in USED_PAIRS and r.ratio < TEXT_MIN
    ]
    assert not failing, failing


@pytest.mark.parametrize("name", THEMES)
def test_theme_defines_every_dusk_variable(name):
    theme = omarchy.build_theme(load_fixture(name))
    for key in DUSK_VARIABLES:
        assert theme.variables[key].startswith("#"), key


def test_mode_drives_dark_flag():
    assert omarchy.build_theme(load_fixture("tokyo-night")).dark is True
    assert omarchy.build_theme(load_fixture("rose-pine")).dark is False


def test_mapping_uses_theme_colors():
    colors = load_fixture("tokyo-night")
    theme = omarchy.build_theme(colors)
    assert theme.background == "#1A1B26"
    assert theme.foreground == "#A9B1D6"
    assert theme.primary == "#7AA2F7"  # accent
    # Without a window-border gradient: accent → yellow.
    assert theme.variables["grad-from"] == "#7AA2F7"
    assert theme.variables["grad-to"] == "#E0AF68"


def test_gradient_prefers_hyprland_border():
    theme = omarchy.build_theme(load_fixture("hackerman"))
    assert theme.variables["grad-from"] == "#26A269"
    assert theme.variables["grad-to"] == "#2EC27E"


def test_parse_gradient_formats():
    assert omarchy.parse_gradient("rgba(26a269ee) rgba(2ec27eee) 45deg") == (
        "#26A269",
        "#2EC27E",
    )
    assert omarchy.parse_gradient("#112233 #445566") == ("#112233", "#445566")
    assert omarchy.parse_gradient(None) == ()


def test_low_contrast_text_is_pushed_toward_foreground():
    colors = dict(load_fixture("rose-pine"))
    theme = omarchy.build_theme(colors)
    tertiary = theme.variables["text-tertiary"]
    assert contrast_ratio(tertiary, str(theme.panel)) >= TEXT_MIN


def test_limbo_toml_overrides_mapping():
    theme = omarchy.build_theme(load_fixture("gruvbox"), {"thinking": "#ff00ff", "nope": "#000000"})
    assert theme.variables["thinking"] == "#FF00FF"


def test_load_theme_missing_returns_none(tmp_path):
    assert omarchy.load_theme(tmp_path) is None
    assert omarchy.current_state(tmp_path) is None


def test_load_theme_reads_state_dir(tmp_path):
    install(tmp_path, "gruvbox")
    theme = omarchy.load_theme(tmp_path)
    assert theme is not None and theme.name == "omarchy"
    state = omarchy.current_state(tmp_path)
    assert state is not None and state.name == "gruvbox"


def test_broken_colors_toml_returns_none(tmp_path):
    (tmp_path / "theme").mkdir()
    (tmp_path / "theme" / "colors.toml").write_text("not = [valid")
    assert omarchy.load_theme(tmp_path) is None


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
async def test_auto_without_omarchy_uses_limbo_dusk(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == DEFAULT_THEME


@pytest.mark.asyncio
async def test_auto_follows_omarchy(tmp_path):
    install(omarchy.state_dir(), "tokyo-night")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "omarchy"
        assert pilot.app.current_theme.background == "#1A1B26"


@pytest.mark.asyncio
async def test_explicit_theme_ignores_omarchy(tmp_path):
    install(omarchy.state_dir(), "tokyo-night")
    app = make_app(tmp_path, theme="limbo-light")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "limbo-light"


@pytest.mark.asyncio
async def test_omarchy_switch_is_picked_up_live(tmp_path):
    state = omarchy.state_dir()
    install(state, "tokyo-night")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        install(state, "rose-pine")
        # Force a distinct mtime even on coarse-grained filesystems.
        later = os.stat(state / "theme" / "colors.toml").st_mtime_ns + 10**9
        os.utime(state / "theme" / "colors.toml", ns=(later, later))
        pilot.app._sync_omarchy_theme()
        await pilot.pause()
        assert pilot.app.current_theme.background == "#FAF4ED"
        assert pilot.app.current_theme.dark is False


@pytest.mark.asyncio
async def test_omarchy_removed_falls_back_to_dusk(tmp_path):
    state = omarchy.state_dir()
    install(state, "gruvbox")
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        shutil.rmtree(state)
        pilot.app._sync_omarchy_theme()
        await pilot.pause()
        assert pilot.app.theme == DEFAULT_THEME
