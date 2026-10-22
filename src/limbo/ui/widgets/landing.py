"""Startup landing (Limbo Dusk): gradient wordmark, environment, next steps.

Mounted at the top of the chat on fresh sessions (``[ui] show_banner``).
The wordmark is lit once by a band of light (Limbo Motion "Dusk Sweep",
1.2s) and then stays still; the ``╱`` fields above and below are static.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from limbo import __version__
from limbo.sessions import SessionMeta
from limbo.ui import motion
from limbo.ui.gradient import theme_gradient

WORDMARK = (
    "██       ██  ███▄   ▄███  ██████▄   ▄█████▄",
    "██       ██  ██ ▀█▄█▀ ██  ██   ██  ██     ██",
    "██       ██  ██   ▀   ██  ██████▀  ██     ██",
    "██       ██  ██       ██  ██   ██  ██     ██",
    "███████  ██  ██       ██  ██████▀   ▀█████▀ ",
)
TAGLINE = "介于想法和提交之间的那个地方。"
SWEEP_FRAMES = 24
_DIAG = "╱" * 400

TIPS = (
    ("/", "命令和技能"),
    ("@ 或路径", "引用文件"),
    ("!", "直接跑 shell"),
    ("ctrl+o", "展开工具输出"),
)


def relative_time(iso: str, now: datetime | None = None) -> str:
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    now = now or datetime.now(timezone.utc)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 60:
        return "刚刚"
    if seconds < 3600:
        return f"{seconds // 60} 分钟前"
    if seconds < 86_400:
        return f"{seconds // 3600} 小时前"
    return f"{seconds // 86_400} 天前"


class Landing(Vertical):
    """Fresh-session landing card."""

    def __init__(
        self,
        model: str,
        detail: str,
        branch: str | None,
        recent: list[SessionMeta],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._detail = detail
        self._branch = branch
        self._recent = recent[:3]
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static(_DIAG, classes="landing-diag", markup=False)
        with Horizontal(classes="landing-head"):
            yield Static(classes="landing-logo", markup=False)
            yield Static(classes="landing-info", markup=False)
        yield Static(_DIAG, classes="landing-diag", markup=False)
        with Horizontal(classes="landing-cols"):
            yield Static(classes="landing-recent", markup=False)
            yield Static(classes="landing-tips", markup=False)

    @property
    def transcript(self) -> str:
        return "\n".join(WORDMARK) + f"\n{TAGLINE}\n{self._model}"

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, lambda _theme: self.refresh_all())
        clock = motion.get_clock(self.app)
        self._frame = 0 if clock.mode == motion.FULL else SWEEP_FRAMES
        self.refresh_all()
        if self._frame < SWEEP_FRAMES:
            clock.subscribe(self, self._on_frame)

    def on_unmount(self) -> None:
        try:
            motion.get_clock(self.app).unsubscribe(self)
        except Exception:  # noqa: BLE001 - app already gone
            pass

    def _on_frame(self, _step: int) -> None:
        self._frame += 1
        self._render_logo()
        if self._frame >= SWEEP_FRAMES:
            motion.get_clock(self.app).unsubscribe(self)

    def refresh_all(self) -> None:
        self._render_logo()
        self._render_info()
        self._render_columns()

    def _colors(self) -> dict[str, str]:
        theme = self.app.current_theme
        v = theme.variables
        return {
            "fg": str(theme.foreground),
            "mid": v.get("text-secondary", str(theme.foreground)),
            "dim": v.get("text-tertiary", str(theme.foreground)),
            "diag": v.get("diag", v.get("text-tertiary", "")),
            "accent": str(theme.accent or theme.primary),
            "primary": str(theme.primary),
            "sel_bg": v.get("selected-bg", str(theme.panel)),
            "sel_fg": v.get("selected-fg", str(theme.foreground)),
        }

    def _render_logo(self) -> None:
        theme = self.app.current_theme
        c = self._colors()
        logo = motion.sweep_lines(
            WORDMARK, theme_gradient(theme), c["diag"], self._frame, total=SWEEP_FRAMES
        )
        self.query_one(".landing-logo", Static).update(logo)
        for diag in self.query(".landing-diag"):
            diag.styles.color = c["diag"]

    def _render_info(self) -> None:
        c = self._colors()
        text = Text()
        text.append("limbo code ", Style(color=c["dim"]))
        text.append(f"v{__version__}\n", Style(color=c["accent"]))
        text.append(TAGLINE + "\n\n", Style(color=c["mid"]))
        text.append("◇ ", Style(color=c["accent"]))
        text.append(self._model, Style(color=c["fg"], bold=True))
        if self._detail:
            text.append(f" · {self._detail}", Style(color=c["dim"]))
        if self._branch:
            text.append("\n⎇ ", Style(color=c["dim"]))
            text.append(self._branch, Style(color=c["fg"]))
        self.query_one(".landing-info", Static).update(text)

    def _render_columns(self) -> None:
        c = self._colors()
        recent = Text("最近会话", Style(color=c["dim"]))
        recent.append("   /sessions 全部\n", Style(color=c["dim"]))
        if not self._recent:
            recent.append("还没有历史会话", Style(color=c["dim"]))
        for i, meta in enumerate(self._recent):
            title = meta.title or "（无标题）"
            when = relative_time(meta.updated_at)
            style = Style(color=c["fg"] if i == 0 else c["mid"])
            recent.append(("▸ " if i == 0 else "  ") + title, style)
            if when:
                recent.append(f"  {when}", Style(color=c["dim"]))
            recent.append("\n")
        recent.rstrip()
        self.query_one(".landing-recent", Static).update(recent)

        tips = Text("从这里开始\n", Style(color=c["dim"]))
        for key, desc in TIPS:
            tips.append(key, Style(color=c["primary"], bold=True))
            tips.append(f"  {desc}\n", Style(color=c["mid"]))
        tips.rstrip()
        self.query_one(".landing-tips", Static).update(tips)
