"""Limbo 主题：limbo-dark / limbo-light。

三层色彩架构（RFC LIM-16 v1.1 §6.2）：

1. 原始色（defs）— 从吉祥物色板派生：蓝 #3069A9 / 黄 #FDE75C / 红 #ED5853 / 深靛 #41415E
2. 语义 token — Theme 槽位（primary/accent/success/warning/error）+ ``variables`` 扩展
3. 组件 CSS — ``app.tcss`` 只引用语义变量，禁止裸 hex

文字色 vs 背景色对比度全部 ≥ 4.5:1（WCAG 相对亮度公式），由
``scripts/check_contrast.py`` 全枚举校验。
"""

from __future__ import annotations

from textual.theme import Theme

DEFAULT_THEME = "limbo-dark"

LIMBO_DARK = Theme(
    name="limbo-dark",
    # 吉祥物蓝的文字级提亮版：running/thinking 状态、链接、prompt 符、active 边框
    primary="#6B9BD2",
    secondary="#9AA0B5",
    # 琥珀橙，仅用于警告/未来的"等待确认"，与 accent 黄拉开
    warning="#E5A04F",
    # 吉祥物衣服红：错误符号/边框/diff 删除前景
    error="#ED5853",
    success="#7EC88F",
    # 吉祥物黄的降饱和文字级版本：选中态、列表符号、次要 highlight
    accent="#E8C85A",
    foreground="#DDE1EC",
    background="#14151F",
    surface="#1B1D2A",
    panel="#232637",
    dark=True,
    variables={
        # 前景阶第二/三级：次要信息、thinking 推理文本、quote/hr
        "text-secondary": "#9AA0B5",
        "text-tertiary": "#7F85A0",
        "border-subtle": "#2C3044",
        "border-default": "#3A3F57",
        # 错误底块上的文字专用（on error-bg = 4.91:1）
        "error-bright": "#F06661",
        "error-bg": "#342026",
        "diff-added-bg": "#1E2E26",
        "diff-removed-bg": "#342026",
    },
)

LIMBO_LIGHT = Theme(
    name="limbo-light",
    primary="#2F5D8F",
    secondary="#4B5265",
    warning="#9A5B00",
    error="#C33C38",
    success="#2E7D4C",
    # on bg-elevated = 5.00:1（选中态文字用在弹窗底上，需 ≥4.5）
    accent="#7A5E0C",
    foreground="#1E2433",
    background="#FAFBFD",
    surface="#F0F2F8",
    panel="#E5E8F2",
    dark=False,
    variables={
        "text-secondary": "#4B5265",
        "text-tertiary": "#6B7186",
        "border-subtle": "#D8DCE8",
        "border-default": "#C4CADD",
        "error-bright": "#A93230",
        "error-bg": "#F3D6D6",
        "diff-added-bg": "#D9EBDD",
        "diff-removed-bg": "#F3D6D6",
    },
)

BUILTIN_THEMES = (LIMBO_DARK, LIMBO_LIGHT)


def palette_snapshot(theme: Theme) -> dict[str, str]:
    """把 Theme 展平成 {token: hex}，供对比度校验脚本全枚举。"""
    snap: dict[str, str] = {
        "primary": str(theme.primary),
        "secondary": str(theme.secondary or theme.primary),
        "warning": str(theme.warning or theme.primary),
        "error": str(theme.error or theme.primary),
        "success": str(theme.success or theme.primary),
        "accent": str(theme.accent or theme.primary),
        "text-primary": str(theme.foreground),
        "bg-base": str(theme.background),
        "bg-surface": str(theme.surface or theme.background),
        "bg-elevated": str(theme.panel or theme.surface or theme.background),
    }
    for key, value in theme.variables.items():
        snap[key] = value
    snap.setdefault("text-secondary", snap["secondary"])
    return snap
