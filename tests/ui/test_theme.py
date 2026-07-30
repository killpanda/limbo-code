"""Tests for the limbo-dark / limbo-light themes (RFC LIM-16 v1.1 P0-1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.theme import DEFAULT_THEME, LIMBO_DARK, LIMBO_LIGHT, palette_snapshot


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover - make this an async generator


def make_app(tmp_path: Path, config: Config | None = None) -> LimboApp:
    cfg = config or Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


@pytest.mark.asyncio
async def test_default_theme_is_limbo_dark(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == DEFAULT_THEME == "limbo-dark"


@pytest.mark.asyncio
async def test_builtin_themes_registered(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        available = pilot.app.available_themes
        assert "limbo-dark" in available
        assert "limbo-light" in available


@pytest.mark.asyncio
async def test_theme_override(tmp_path):
    cfg = Config()
    cfg.ui.theme = "limbo-light"
    app = make_app(tmp_path, config=cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "limbo-light"


@pytest.mark.asyncio
async def test_theme_override_to_textual_builtin(tmp_path):
    cfg = Config()
    cfg.ui.theme = "nord"
    app = make_app(tmp_path, config=cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "nord"


@pytest.mark.asyncio
async def test_unknown_theme_falls_back_to_limbo_dark(tmp_path):
    cfg = Config()
    cfg.ui.theme = "no-such-theme"
    app = make_app(tmp_path, config=cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert pilot.app.theme == "limbo-dark"


def test_palette_snapshot_contains_semantic_tokens():
    snap = palette_snapshot(LIMBO_DARK)
    for token in (
        "primary",
        "accent",
        "success",
        "warning",
        "error",
        "text-primary",
        "text-secondary",
        "text-tertiary",
        "bg-base",
        "bg-surface",
        "bg-elevated",
        "error-bright",
        "error-bg",
    ):
        assert token in snap, token
        assert snap[token].startswith("#"), token


def test_light_palette_differs_from_dark():
    dark = palette_snapshot(LIMBO_DARK)
    light = palette_snapshot(LIMBO_LIGHT)
    assert dark["bg-base"] != light["bg-base"]
    assert dark["text-primary"] != light["text-primary"]
