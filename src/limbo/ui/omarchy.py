"""跟随 Omarchy 系统主题。

Omarchy 把当前主题放在 ``~/.local/state/omarchy/current/theme/``，每个主题都带
一份 ``colors.toml``（accent / background / foreground / 八个 ANSI 色及其 bright
版本 / mode）。这里直接读它，按固定映射算出一套 Textual 主题，不依赖 Omarchy
的脚本，也不要求它渲染任何 limbo 专用模板。

可选覆盖：当前主题目录里存在 ``limbo.toml``（通常由用户放在
``~/.config/omarchy/themed/limbo.toml.tpl`` 的模板渲染而来）时，其中与 limbo
token 同名的键会覆盖自动映射的结果。

映射完成后统一做对比度兜底：文字类 token 在三级背景上不足 4.5:1 时，逐步向
foreground 混色直到达标，浅色主题上的灰字也能读清。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from textual.theme import Theme

from limbo.ui.contrast import readable
from limbo.ui.gradient import mix

THEME_NAME = "omarchy"
DEFAULT_STATE_DIR = Path("~/.local/state/omarchy/current").expanduser()
_HEX = re.compile(r"^#?([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?$")
_RGBA = re.compile(r"rgba?\(\s*([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})?\s*\)")

# 背景上的文字 token：都要对三级背景达到 TEXT_MIN。
_TEXT_TOKENS = (
    "text-secondary",
    "text-tertiary",
    "user-message-fg",
    "thinking",
)
_SLOT_TEXT = ("primary", "accent", "success", "warning", "error")


@dataclass(frozen=True)
class OmarchyState:
    """当前 Omarchy 主题的文件指纹，用来判断是否需要重建主题。"""

    name: str
    mtime_ns: int


def state_dir() -> Path:
    return DEFAULT_STATE_DIR


def colors_path(base: Path | None = None) -> Path:
    return (base or state_dir()) / "theme" / "colors.toml"


def is_available(base: Path | None = None) -> bool:
    return colors_path(base).is_file()


def current_state(base: Path | None = None) -> OmarchyState | None:
    """主题名 + colors.toml 的 mtime；Omarchy 不存在时返回 None。"""
    root = base or state_dir()
    path = colors_path(root)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    try:
        name = (root / "theme.name").read_text(encoding="utf-8").strip()
    except OSError:
        name = ""
    return OmarchyState(name=name, mtime_ns=mtime)


def _normalize(color: object) -> str | None:
    if not isinstance(color, str):
        return None
    match = _HEX.match(color.strip())
    if match:
        return "#" + match.group(1).upper()
    return None


def parse_gradient(spec: object) -> tuple[str, ...]:
    """解析 ``hyprland_active_border`` 这类 ``rgba(26a269ee) rgba(2ec27eee) 45deg``。"""
    if not isinstance(spec, str):
        return ()
    colors = ["#" + m.group(1).upper() for m in _RGBA.finditer(spec)]
    if not colors:
        colors = [c for c in (_normalize(part) for part in spec.split()) if c]
    return tuple(colors)


def build_theme(colors: dict[str, object], overrides: dict[str, object] | None = None) -> Theme:
    """按映射规则把 Omarchy 的 colors.toml 转成 Textual 主题。"""

    def pick(key: str, *fallbacks: str) -> str:
        for name in (key, *fallbacks):
            value = _normalize(colors.get(name))
            if value:
                return value
        raise KeyError(key)

    bg = pick("background", "color0")
    fg = pick("foreground", "color7")
    accent = pick("accent", "blue", "color4")
    yellow = pick("yellow", "color3")
    red = pick("red", "color1")
    green = pick("green", "color2")
    magenta = pick("magenta", "color5")
    orange = pick("orange", "yellow", "color3")
    dark = str(colors.get("mode", "dark")).lower() != "light"

    surface = mix(bg, fg, 0.05)
    panel = mix(bg, fg, 0.10)
    backgrounds = (bg, surface, panel)
    # 文字不达标时往更远离背景的方向推：暗色主题推向前景，浅色同理。
    toward = fg

    gradient = parse_gradient(colors.get("hyprland_active_border"))
    if len(gradient) < 2:
        gradient = (accent, yellow)
    grad_from, grad_to = gradient[0], gradient[-1]

    user_bg = mix(bg, accent, 0.12)
    variables = {
        "text-secondary": mix(fg, bg, 0.34),
        "text-tertiary": mix(fg, bg, 0.52),
        "border-subtle": mix(bg, fg, 0.16),
        "border-default": mix(bg, fg, 0.28),
        "user-message-bg": user_bg,
        "user-message-fg": mix(fg, accent, 0.45),
        "error-bright": red,
        "error-bg": mix(bg, red, 0.15),
        "diff-added": green,
        "diff-added-bg": mix(bg, green, 0.15),
        "diff-removed": red,
        "diff-removed-bg": mix(bg, red, 0.15),
        "diag": mix(bg, fg, 0.28),
        "thinking": magenta,
        "selected-bg": mix(bg, accent, 0.25),
        "selected-fg": fg,
        "grad-from": grad_from,
        "grad-mid": mix(grad_from, grad_to, 0.5),
        "grad-to": grad_to,
    }
    slots = {
        "primary": accent,
        "accent": yellow,
        "success": green,
        "warning": orange,
        "error": red,
    }

    # Contrast fixes apply to derived colours only: a value the user pinned
    # in limbo.toml is taken as-is.
    for key in _SLOT_TEXT:
        slots[key] = readable(slots[key], backgrounds, toward)
    for key in _TEXT_TOKENS:
        bgs = (user_bg,) if key == "user-message-fg" else backgrounds
        variables[key] = readable(variables[key], bgs, toward)
    for key, bg_key in (
        ("error-bright", "error-bg"),
        ("diff-added", "diff-added-bg"),
        ("diff-removed", "diff-removed-bg"),
        ("selected-fg", "selected-bg"),
    ):
        variables[key] = readable(variables[key], (variables[bg_key],), toward)

    for key, value in (overrides or {}).items():
        normalized = _normalize(value)
        if not normalized:
            continue
        if key in slots:
            slots[key] = normalized
        elif key in variables:
            variables[key] = normalized

    return Theme(
        name=THEME_NAME,
        primary=slots["primary"],
        secondary=variables["text-secondary"],
        warning=slots["warning"],
        error=slots["error"],
        success=slots["success"],
        accent=slots["accent"],
        foreground=fg,
        background=bg,
        surface=surface,
        panel=panel,
        dark=dark,
        variables=variables,
    )


def _read_toml(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def load_theme(base: Path | None = None) -> Theme | None:
    """读当前 Omarchy 主题；不存在或格式不对时返回 None（调用方回落到 limbo-dusk）。"""
    root = base or state_dir()
    colors = _read_toml(colors_path(root))
    if colors is None:
        return None
    overrides = _read_toml(root / "theme" / "limbo.toml")
    try:
        return build_theme(colors, overrides)
    except KeyError:
        return None
