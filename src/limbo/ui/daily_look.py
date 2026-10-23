"""Daily mode 外观 Limbo Dawn：当前主题的暖色变体（设计稿 daily-mode-ui）。

Dusk 是黄昏（蓝 → 黄），Dawn 是日出（红 → 杏橙 → 黄）。布局、``╱`` 纹理、
吉祥物黄和语义色（成功/错误/diff）都不变，只换色温：primary、渐变、用户消息
染色、选中底和三级背景。

变体是一个独立注册的 Textual 主题，名字为 ``<基础主题>+daily``。组件都订阅了
``theme_changed_signal`` 并从主题变量取色，所以切换外观只需要换 ``app.theme``。

- limbo 自带的三个主题用手调的 Dawn 色板（``DAWN_PALETTES``）
- 其他主题（Omarchy 生成的、Textual 内置的）按规则派生：渐变取
  error → warning → accent，primary 取 warning，用户消息底往 warning 混色；
  文字类颜色不足 4.5:1 时向前景推，和 Omarchy 映射的兜底一致
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from textual.theme import Theme

from limbo.ui.contrast import readable
from limbo.ui.gradient import mix

DAILY_SUFFIX = "+daily"
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

# Theme 槽位（primary/background/…）和 variables 分开存，方便 replace()。
_DAWN_DARK: dict[str, Any] = {
    "slots": {
        "primary": "#F5A06E",
        "foreground": "#EEE6E1",
        "background": "#15121A",
        "surface": "#1C1822",
        "panel": "#262029",
        "secondary": "#B9ADB4",
    },
    "variables": {
        "text-secondary": "#B9ADB4",
        "text-tertiary": "#8E8290",
        "border-subtle": "#2E2733",
        "border-default": "#463B4E",
        "user-message-bg": "#2C1F27",
        "user-message-fg": "#F6C9A6",
        "diag": "#43384A",
        "thinking": "#E6A2C4",
        "selected-bg": "#4A2E35",
        "selected-fg": "#EEE6E1",
        # 日出渐变：吉祥物红 → 杏橙 → 吉祥物黄
        "grad-from": "#ED5853",
        "grad-mid": "#F5A06E",
        "grad-to": "#FDE75C",
    },
}

# limbo-dark 本身比 dusk 亮一档，Dawn 的三级背景跟着抬一点。
_DAWN_DARK_SOFT: dict[str, Any] = {
    "slots": {
        **_DAWN_DARK["slots"],
        "background": "#18151D",
        "surface": "#1F1A25",
        "panel": "#29222E",
    },
    "variables": dict(_DAWN_DARK["variables"]),
}

_DAWN_LIGHT: dict[str, Any] = {
    "slots": {
        "primary": "#A54A1C",
        "foreground": "#2A1F24",
        "background": "#FCF8F4",
        "surface": "#F5EEE7",
        "panel": "#EDE3DA",
        "secondary": "#5A4A50",
        "success": "#2A7346",
    },
    "variables": {
        "text-secondary": "#5A4A50",
        "text-tertiary": "#76646B",
        "border-subtle": "#E6D9CD",
        "border-default": "#D6C4B5",
        "user-message-bg": "#F7E1D2",
        "user-message-fg": "#8A3A12",
        "diag": "#E0D2C6",
        "thinking": "#8A3D66",
        "selected-bg": "#F0D3C2",
        "selected-fg": "#2A1F24",
        "grad-from": "#C33C38",
        "grad-mid": "#A54A1C",
        "grad-to": "#9A7A0C",
    },
}

DAWN_PALETTES: dict[str, dict[str, Any]] = {
    "limbo-dusk": _DAWN_DARK,
    "limbo-dark": _DAWN_DARK_SOFT,
    "limbo-light": _DAWN_LIGHT,
}


def variant_name(base_name: str) -> str:
    return base_name + DAILY_SUFFIX


def base_name(name: str) -> str:
    """``limbo-dusk+daily`` → ``limbo-dusk``；不是变体名时原样返回。"""
    return name.removesuffix(DAILY_SUFFIX)


def _is_hex(color: object) -> bool:
    return isinstance(color, str) and bool(_HEX.match(color))


def derive_dawn(base: Theme) -> dict[str, Any]:
    """非 limbo 主题的 Dawn 覆盖项（slots + variables）。

    ANSI 主题（``ansi_red`` 这类终端色名）没法混色，不改颜色，只靠徽标区分。
    """
    colors = (base.background, base.foreground, base.warning, base.error, base.accent)
    if not all(c is None or _is_hex(c) for c in colors) or base.ansi:
        return {"slots": {}, "variables": {}}
    dark = base.dark
    bg = str(base.background or ("#121212" if dark else "#EFEFEF"))
    fg = str(base.foreground or ("#E0E0E0" if dark else "#1E1E1E"))
    surface = str(base.surface or mix(bg, fg, 0.05))
    panel = str(base.panel or mix(bg, fg, 0.10))
    backgrounds = (bg, surface, panel)
    warm = str(base.warning or base.accent or base.primary)
    red = str(base.error or warm)
    yellow = str(base.accent or warm)
    user_bg = mix(bg, warm, 0.14)
    selected_bg = mix(bg, warm, 0.25)
    return {
        "slots": {"primary": readable(warm, backgrounds, fg)},
        "variables": {
            "user-message-bg": user_bg,
            "user-message-fg": readable(mix(fg, warm, 0.45), (user_bg,), fg),
            "selected-bg": selected_bg,
            "selected-fg": readable(fg, (selected_bg,), fg),
            "grad-from": red,
            "grad-mid": warm,
            "grad-to": yellow,
        },
    }


def daily_variant(base: Theme) -> Theme:
    """``base`` 的 daily 外观主题，名字为 ``<base.name>+daily``。"""
    palette = DAWN_PALETTES.get(base.name) or derive_dawn(base)
    return replace(
        base,
        name=variant_name(base.name),
        variables={**base.variables, **palette["variables"]},
        **palette["slots"],
    )
