"""Sidebar + responsive layout (Limbo Dusk design, step 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.sidebar import (
    WIDE_MIN_WIDTH,
    Sidebar,
    count_changes,
    format_tokens,
    git_branch,
)
from limbo.ui.widgets.status_bar import StatusBar


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover


def make_app(tmp_path: Path) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


def test_count_changes_edit_and_write():
    assert count_changes("edit", {"path": "a.py", "old_text": "x\ny", "new_text": "z"}) == (
        "a.py",
        1,
        2,
    )
    assert count_changes("write", {"path": "b.py", "content": "1\n2\n3\n"}) == ("b.py", 3, 0)
    assert count_changes("read", {"path": "a.py"}) is None
    assert count_changes("edit", {"old_text": "x"}) is None


def test_git_branch(tmp_path):
    assert git_branch(tmp_path) is None
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/feat/x\n")
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    assert git_branch(nested) == "feat/x"
    (tmp_path / ".git" / "HEAD").write_text("0123456789abcdef\n")
    assert git_branch(tmp_path) == "01234567"


def test_format_tokens():
    assert format_tokens(950) == "950"
    assert format_tokens(1500) == "1.5K"
    assert format_tokens(412_000) == "412K"
    assert format_tokens(1_000_000) == "1.0M"


@pytest.mark.asyncio
async def test_wide_terminal_shows_sidebar(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(WIDE_MIN_WIDTH + 20, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        assert not screen.has_class("-compact")
        assert screen.query_one("#sidebar", Sidebar).display


@pytest.mark.asyncio
async def test_narrow_terminal_hides_sidebar_and_ctrl_d_peeks(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(WIDE_MIN_WIDTH - 20, 30)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        sidebar = screen.query_one("#sidebar", Sidebar)
        assert screen.has_class("-compact")
        assert sidebar.styles.display == "none"
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert screen.has_class("-peek")
        assert sidebar.styles.display == "block"
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert not screen.has_class("-peek")


@pytest.mark.asyncio
async def test_resize_switches_layout(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(WIDE_MIN_WIDTH + 10, 40)) as pilot:
        await pilot.pause()
        assert not pilot.app.screen.has_class("-compact")
        await pilot.resize_terminal(WIDE_MIN_WIDTH - 10, 40)
        await pilot.pause()
        assert pilot.app.screen.has_class("-compact")


@pytest.mark.asyncio
async def test_tool_results_feed_changed_files(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        sidebar = screen.query_one("#sidebar", Sidebar)
        sidebar.record_tool_result(
            "edit", {"path": str(tmp_path / "a.py"), "old_text": "x", "new_text": "y\nz"}
        )
        sidebar.record_tool_result("write", {"path": str(tmp_path / "b.py"), "content": "1"})
        sidebar.record_tool_result("read", {"path": str(tmp_path / "a.py")})
        assert sidebar.file_totals == (2, 3, 1)
        screen._sync_usage()
        assert "+3−1" in screen.query_one("#statusbar", StatusBar)._context_text()


@pytest.mark.asyncio
async def test_usage_drives_context_percent(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        sidebar = pilot.app.screen.query_one("#sidebar", Sidebar)
        sidebar.set_usage(5000, 50_000, 200_000)
        assert sidebar.context_percent == 25
        sidebar.set_usage(5000, 500_000, 200_000)
        assert sidebar.context_percent == 100


@pytest.mark.asyncio
async def test_new_session_resets_sidebar(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        sidebar = screen.query_one("#sidebar", Sidebar)
        sidebar.record_tool_result("write", {"path": "x.py", "content": "1"})
        screen._start_new_session()
        await pilot.pause()
        assert sidebar.file_totals == (0, 0, 0)
