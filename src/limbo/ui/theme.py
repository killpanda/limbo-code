"""Limbo 主题：limbo-dusk（默认）/ limbo-dark / limbo-light，外加运行时从
Omarchy 生成的 ``omarchy`` 主题（见 ``limbo.ui.omarchy``）。

三层色彩架构（RFC LIM-16 v1.1 §6.2）：

1. 原始色（defs）— 从吉祥物色板派生：蓝 #3069A9 / 黄 #FDE75C / 红 #ED5853 / 深靛 #41415E
2. 语义 token — Theme 槽位（primary/accent/success/warning/error）+ ``variables`` 扩展
3. 组件 CSS — ``app.tcss`` 只引用语义变量，禁止裸 hex

文字色 vs 背景色对比度全部 ≥ 4.5:1（WCAG 相对亮度公式），由
``scripts/check_contrast.py`` 全枚举校验。
"""

from __future__ import annotations

from textual.theme import Theme

DEFAULT_THEME = "limbo-dusk"

# Limbo Dusk 在 Textual 标准槽位之外新增的语义变量。每个 limbo 主题（以及
# Omarchy 生成的主题）都必须给出全部取值，app.tcss 和渐变渲染只认这些名字。
#   diag        —— ``╱`` 斜线纹理
#   thinking    —— 推理文本
#   selected-bg / selected-fg —— 列表选中行、激活标签
#   grad-from / grad-mid / grad-to —— 字标、上下文进度条、主按钮的渐变
DUSK_VARIABLES = (
    "diag",
    "thinking",
    "selected-bg",
    "selected-fg",
    "grad-from",
    "grad-mid",
    "grad-to",
)

LIMBO_DUSK = Theme(
    name="limbo-dusk",
    # 吉祥物蓝提亮：运行中、焦点边框、提示符
    primary="#5B9BF0",
    secondary="#A4A9C0",
    warning="#F0A35A",
    error="#ED5853",
    success="#6FD6A8",
    # 吉祥物黄：选中项、思考强度、数字强调
    accent="#FDE75C",
    foreground="#E3E5EF",
    background="#11121C",
    surface="#171928",
    panel="#1F2235",
    dark=True,
    variables={
        "text-secondary": "#A4A9C0",
        "text-tertiary": "#7C82A3",
        "border-subtle": "#2A2D44",
        "border-default": "#3C4060",
        "user-message-bg": "#1C2340",
        "user-message-fg": "#A9C6F2",
        "error-bright": "#F2706B",
        "error-bg": "#331D24",
        "diff-added": "#6FD6A8",
        "diff-added-bg": "#17291F",
        "diff-removed": "#F2706B",
        "diff-removed-bg": "#331D24",
        "diag": "#3C4060",
        "thinking": "#A48CFF",
        "selected-bg": "#2A3F66",
        "selected-fg": "#E3E5EF",
        # 黄昏渐变：深蓝 → 天蓝 → 黄
        "grad-from": "#3069A9",
        "grad-mid": "#5B9BF0",
        "grad-to": "#FDE75C",
    },
)

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
        # 用户消息专属蓝：蓝染底 + 蓝字，与 agent 中性前景拉开辨识度
        # （user-fg on user-bg = 7.26:1，on chat-bg = 7.91:1）
        "user-message-bg": "#202744",
        "user-message-fg": "#96B9EA",
        # 错误底块上的文字专用（on error-bg = 4.91:1）
        "error-bright": "#F06661",
        "error-bg": "#342026",
        # diff 专用前景/底色（Generic.Inserted/Deleted 渲染在此底上）
        "diff-added": "#7EC88F",
        "diff-added-bg": "#1E2E26",
        "diff-removed": "#F06661",
        "diff-removed-bg": "#342026",
        "diag": "#3A3F57",
        "thinking": "#A99BE0",
        "selected-bg": "#2A3756",
        "selected-fg": "#DDE1EC",
        "grad-from": "#3069A9",
        "grad-mid": "#6B9BD2",
        "grad-to": "#E8C85A",
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
        # 用户消息专属蓝：淡蓝底 + 深蓝字，与 agent 中性前景拉开辨识度
        # （user-fg on user-bg = 6.69:1，on chat-bg = 8.78:1）
        "user-message-bg": "#DCE6F5",
        "user-message-fg": "#1F4E86",
        "error-bright": "#A93230",
        "error-bg": "#F3D6D6",
        "diff-added": "#256B41",
        "diff-added-bg": "#D9EBDD",
        "diff-removed": "#A93230",
        "diff-removed-bg": "#F3D6D6",
        "diag": "#C4CADD",
        "thinking": "#5B4A9E",
        "selected-bg": "#D3DEF0",
        "selected-fg": "#1E2433",
        "grad-from": "#2F5D8F",
        "grad-mid": "#3069A9",
        "grad-to": "#9A7A0C",
    },
)

BUILTIN_THEMES = (LIMBO_DUSK, LIMBO_DARK, LIMBO_LIGHT)


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
