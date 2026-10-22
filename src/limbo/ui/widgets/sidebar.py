"""Right-hand sidebar (Limbo Dusk): session, model, context, changed files.

Always visible on terminals at least ``WIDE_MIN_WIDTH`` columns wide; on
narrower ones the main screen hides it and the top status bar carries a
compact summary instead (``ctrl+d`` peeks the sidebar as an overlay).

The sidebar is a pure view: the main screen pushes state in through the
``set_*`` / ``record_*`` methods; nothing here reads the agent directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from limbo.ui.gradient import gradient_text, ramp, theme_gradient

WIDE_MIN_WIDTH = 120
SIDEBAR_WIDTH = 36
# Usable text columns: width minus 2×2 padding, the left border and a
# column kept free for the scrollbar.
INNER_WIDTH = SIDEBAR_WIDTH - 6
_METER_CELLS = 10
_MAX_FILES = 8


def git_branch(workdir: Path) -> str | None:
    """Current branch from ``.git/HEAD`` (no subprocess; None outside git)."""
    for directory in (workdir, *workdir.parents):
        head = directory / ".git" / "HEAD"
        if head.is_file():
            try:
                ref = head.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if ref.startswith("ref: refs/heads/"):
                return ref.removeprefix("ref: refs/heads/")
            return ref[:8] or None
    return None


def short_path(path: str, home: Path | None = None) -> str:
    home = home or Path.home()
    try:
        return "~/" + str(Path(path).resolve().relative_to(home))
    except ValueError:
        return path


def format_tokens(total: int) -> str:
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    if total >= 1000:
        return f"{total / 1000:.0f}K" if total >= 10_000 else f"{total / 1000:.1f}K"
    return str(total)


def _clip(value: str, width: int) -> str:
    return value if len(value) <= width else "…" + value[-(width - 1):]


@dataclass
class FileChange:
    added: int = 0
    removed: int = 0


def count_changes(tool: str, arguments: dict[str, Any]) -> tuple[str, int, int] | None:
    """Line delta of a successful edit/write call, or None for other tools."""
    path = arguments.get("path")
    if not isinstance(path, str) or not path:
        return None
    if tool == "edit":
        old = arguments.get("old_text") or ""
        new = arguments.get("new_text") or ""
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        return path, len(new.splitlines()), len(old.splitlines())
    if tool == "write":
        content = arguments.get("content") or ""
        if not isinstance(content, str):
            return None
        return path, len(content.splitlines()), 0
    return None


class Sidebar(VerticalScroll):
    """Session details column docked on the right of the chat."""

    can_focus = False

    def __init__(self, model: str, workdir: Path, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._provider = ""
        self._thinking: str | None = None
        self._workdir = workdir
        self._title = ""
        self._tokens = 0
        self._context_tokens = 0
        self._context_window = 0
        self._files: dict[str, FileChange] = {}
        self._goal: str | None = None
        self._goal_progress: tuple[int, int] | None = None
        self._code_mode = False
        self._plan_mode = False
        self._skills: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static(id="sb-logo", markup=False)
        yield Static(id="sb-session", classes="sb-group", markup=False)
        yield Static(id="sb-model", classes="sb-group", markup=False)
        yield Static(id="sb-files", classes="sb-group", markup=False)
        yield Static(id="sb-goal", classes="sb-group", markup=False)
        yield Static(id="sb-extra", classes="sb-group", markup=False)

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_all())
        self.refresh_all()

    # -- state setters ---------------------------------------------------------

    def set_model(self, model: str, provider: str = "", thinking: str | None = None) -> None:
        self._model, self._provider, self._thinking = model, provider, thinking
        self._render_model()

    def set_title(self, title: str) -> None:
        if title != self._title:
            self._title = title
            self._render_session()

    def set_usage(self, total_tokens: int, context_tokens: int, context_window: int) -> None:
        self._tokens = total_tokens
        self._context_tokens = context_tokens
        self._context_window = context_window
        self._render_model()

    @property
    def context_percent(self) -> int | None:
        if not self._context_window:
            return None
        return min(100, round(100 * self._context_tokens / self._context_window))

    def record_tool_result(self, tool: str, arguments: dict[str, Any]) -> None:
        delta = count_changes(tool, arguments)
        if delta is None:
            return
        path, added, removed = delta
        try:
            key = str(Path(path).resolve().relative_to(self._workdir.resolve()))
        except ValueError:
            key = path
        change = self._files.setdefault(key, FileChange())
        change.added += added
        change.removed += removed
        self._render_files()

    @property
    def file_totals(self) -> tuple[int, int, int]:
        """(files, lines added, lines removed) for the compact header."""
        added = sum(c.added for c in self._files.values())
        removed = sum(c.removed for c in self._files.values())
        return len(self._files), added, removed

    def set_goal(self, goal: str | None, progress: tuple[int, int] | None = None) -> None:
        self._goal, self._goal_progress = goal, progress
        self._render_goal()

    def set_code_mode(self, enabled: bool) -> None:
        self._code_mode = enabled
        self._render_extra()

    def set_plan_mode(self, enabled: bool) -> None:
        self._plan_mode = enabled
        self._render_extra()

    def set_skills(self, names: list[str]) -> None:
        self._skills = names
        self._render_extra()

    def reset_session(self) -> None:
        """New or resumed session: per-session state starts over."""
        self._title = ""
        self._tokens = self._context_tokens = 0
        self._files.clear()
        self.refresh_all()

    # -- rendering -------------------------------------------------------------

    def _colors(self) -> dict[str, str]:
        theme = self.app.current_theme
        v = theme.variables
        return {
            "fg": str(theme.foreground),
            "dim": v.get("text-tertiary", str(theme.foreground)),
            "mid": v.get("text-secondary", str(theme.foreground)),
            "diag": v.get("diag", v.get("text-tertiary", "")),
            "accent": str(theme.accent or theme.primary),
            "primary": str(theme.primary),
            "success": str(theme.success or theme.primary),
            "error": str(theme.error or theme.primary),
            "warning": str(theme.warning or theme.primary),
        }

    def _update(self, widget_id: str, content: Text) -> None:
        try:
            self.query_one(f"#{widget_id}", Static).update(content)
        except Exception:  # noqa: BLE001 - not composed yet
            pass

    def refresh_all(self) -> None:
        self._render_logo()
        self._render_session()
        self._render_model()
        self._render_files()
        self._render_goal()
        self._render_extra()

    def _heading(self, label: str) -> Text:
        return Text(label + "\n", style=self._colors()["dim"])

    def _render_logo(self) -> None:
        c = self._colors()
        stops = theme_gradient(self.app.current_theme)
        text = Text("Limbo™ ", style=c["dim"])
        text.append_text(gradient_text("LIMBO", stops, True))
        text.append(" " + "╱" * (INNER_WIDTH - text.cell_len - 1), style=c["diag"])
        self._update("sb-logo", text)

    def _render_session(self) -> None:
        c = self._colors()
        text = Text()
        text.append((self._title or "新会话") + "\n", style=f"bold {c['fg']}")
        text.append(_clip(short_path(str(self._workdir)), INNER_WIDTH), style=c["dim"])
        branch = git_branch(self._workdir)
        if branch:
            text.append("\n" + _clip(f"⎇ {branch}", INNER_WIDTH), style=c["dim"])
        self._update("sb-session", text)

    def _render_model(self) -> None:
        c = self._colors()
        text = self._heading("模型")
        text.append("◇ ", style=c["accent"])
        text.append(self._model, style=f"bold {c['fg']}")
        if self._provider:
            text.append(f"  {self._provider}", style=c["dim"])
        text.append("\n")
        if self._thinking:
            text.append("  思考 ", style=c["mid"])
            text.append(self._thinking, style=c["accent"])
            text.append("\n")
        pct = self.context_percent
        if pct is not None:
            filled = round(pct / 100 * _METER_CELLS)
            stops = self._meter_stops(pct)
            text.append("  ")
            for i in range(_METER_CELLS):
                if i < filled:
                    text.append("▰", style=ramp(stops, i / (_METER_CELLS - 1)))
                else:
                    text.append("▱", style=c["diag"])
            text.append(f" {pct}%", style=c["mid"])
            text.append(
                f" {format_tokens(self._context_tokens)}/{format_tokens(self._context_window)}",
                style=c["dim"],
            )
            text.append("\n")
        if self._tokens:
            text.append(f"  累计 {format_tokens(self._tokens)} tokens", style=c["dim"])
        self._update("sb-model", text)

    def _meter_stops(self, pct: int) -> tuple[str, ...]:
        stops = theme_gradient(self.app.current_theme)
        if pct > 80:
            # Past 80% the far end warms up: time to think about /compact.
            return (*stops[:-1], self._colors()["warning"])
        return stops

    def _render_files(self) -> None:
        c = self._colors()
        text = self._heading("修改的文件")
        if not self._files:
            text.append("  还没有改动", style=c["dim"])
        items = sorted(self._files.items(), key=lambda kv: -(kv[1].added + kv[1].removed))
        for path, change in items[:_MAX_FILES]:
            stats = Text()
            if change.added:
                stats.append(f"+{change.added}", style=c["success"])
            if change.removed:
                if change.added:
                    stats.append(" ")
                stats.append(f"−{change.removed}", style=c["error"])
            room = INNER_WIDTH - stats.cell_len - 1
            name = path if len(path) <= room else "…" + path[-(room - 1):]
            text.append(name.ljust(room) + " ", style=c["fg"])
            text.append_text(stats)
            text.append("\n")
        if len(items) > _MAX_FILES:
            text.append(f"  另有 {len(items) - _MAX_FILES} 个文件", style=c["dim"])
        text.rstrip()
        self._update("sb-files", text)

    def _render_goal(self) -> None:
        widget_id = "sb-goal"
        try:
            widget = self.query_one(f"#{widget_id}", Static)
        except Exception:  # noqa: BLE001 - not composed yet
            return
        widget.display = self._goal is not None
        if self._goal is None:
            return
        c = self._colors()
        text = self._heading("目标 /goal")
        text.append("◐ ", style=c["warning"])
        text.append(self._goal, style=c["mid"])
        if self._goal_progress:
            done, total = self._goal_progress
            text.append(f"\n  第 {done}/{total} 轮", style=c["dim"])
        widget.update(text)

    def _render_extra(self) -> None:
        c = self._colors()
        text = self._heading("模式 · 技能")
        text.append("Code Mode ", style=c["mid"])
        if self._code_mode:
            text.append("on", style=c["accent"])
        else:
            text.append("off", style=c["dim"])
        if self._plan_mode:
            text.append(" · 计划模式 ", style=c["mid"])
            text.append("on", style=c["accent"])
        if self._skills:
            shown = ", ".join(self._skills[:4])
            more = f" 等 {len(self._skills)} 个" if len(self._skills) > 4 else ""
            text.append("\n" + shown + more, style=c["mid"])
        self._update("sb-extra", text)
