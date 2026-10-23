"""Snapshot tests for the main UI states (RFC LIM-16 v1.1 Test Plan §7.1).

Four scenarios: fresh idle screen, thinking + running tool card, tool error
+ structured error message, session picker modal. Run
``pytest --snapshot-update`` after intentional visual changes and review the
SVG diff.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from limbo.config import Config
from limbo.ui.app import LimboApp
from limbo.ui.widgets.chat import ChatWidget


class FakeLLMClient:
    async def chat(self, messages, tools, on_request=None):
        return
        yield  # pragma: no cover - make this an async generator


# Fixed workdir so the status bar renders identical text across runs
# (tmp_path contains a random pytest-N segment that would break snapshots).
SNAPSHOT_WORKDIR = Path("/tmp/limbo-snapshot-workdir")


@pytest.fixture(autouse=True)
def _deterministic_ui(monkeypatch):
    """Freeze widget clocks, spinner timer, and workdir display for stable snapshots.

    - Replaces the ``time`` *name inside the widget modules'* namespaces (not
      the global time module — freezing that would stall Textual's own clock).
    - ``limbo.ui.app.Path`` is swapped for a Path whose ``resolve()`` is the
      identity: ``LimboApp`` resolves the workdir, which reads /private/tmp/...
      on macOS but /tmp/... on Linux CI and breaks cross-platform snapshots.
    """
    SNAPSHOT_WORKDIR.mkdir(parents=True, exist_ok=True)
    frozen = SimpleNamespace(monotonic=lambda: 100.0)
    monkeypatch.setattr("limbo.ui.widgets.status_bar.time", frozen)
    monkeypatch.setattr("limbo.ui.widgets.tool_card.time", frozen)
    monkeypatch.setattr("limbo.ui.widgets.activity.time", frozen)
    # Relative times ("3 天前") on the landing and in the session picker are
    # measured against a fixed "now" so snapshots don't age day by day.
    import limbo.ui.widgets.landing as landing_module

    class _FrozenDatetime(landing_module.datetime):  # type: ignore[misc]
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls(2026, 8, 1, 12, 0, tzinfo=tz)

    monkeypatch.setattr(landing_module, "datetime", _FrozenDatetime)
    # __version__ comes from git (vcs versioning): v0.2.2 on a tag,
    # v0.2.3.devN+g<sha> on a branch checkout in CI.
    monkeypatch.setattr(landing_module, "__version__", "0.0.0")
    from limbo.ui.widgets.status_bar import StatusBar

    monkeypatch.setattr(StatusBar, "_tick", lambda self: None)

    class _NoResolvePath(type(Path())):
        def resolve(self, strict: bool = False):  # type: ignore[override]
            return self

    monkeypatch.setattr("limbo.ui.app.Path", _NoResolvePath)
    # The sidebar lists discovered skills, which include ~/.limbo/skills.
    monkeypatch.setattr("limbo.ui.screens.main.discover_skills", lambda workdir, **_: [])


def make_app(tmp_path: Path) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=SNAPSHOT_WORKDIR,
        config=cfg,
        llm_client=FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


def test_snapshot_idle(snap_compare, tmp_path):
    app = make_app(tmp_path)
    assert snap_compare(app, terminal_size=(90, 28))


def test_snapshot_wide_layout_with_sidebar(snap_compare, tmp_path):
    """Wide terminal: chat column beside the session sidebar (Limbo Dusk)."""
    app = make_app(tmp_path)

    async def run_before(pilot) -> None:
        await pilot.pause()
        screen = pilot.app.screen
        chat = screen.query_one("#chat", ChatWidget)
        chat.add_user_message("把 provider 解析抽成独立模块")
        card = chat.add_tool_card(
            "c1", "edit", {"path": "src/limbo/config.py", "old_text": "a\nb", "new_text": "c"}
        )
        card.set_success("ok")
        sidebar = screen.query_one("#sidebar")
        sidebar.record_tool_result(
            "edit", {"path": "src/limbo/config.py", "old_text": "a\nb", "new_text": "c"}
        )
        sidebar.set_usage(12_000, 42_000, 128_000)
        await pilot.pause()

    assert snap_compare(app, terminal_size=(140, 32), run_before=run_before)


