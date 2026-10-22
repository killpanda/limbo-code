"""工具卡片代码高亮的 Pygments 样式，色值与 limbo 主题色板一致（RFC §6.2）。

替代原先硬编码的 ``Syntax(theme="ansi_dark")``，让语法高亮纳入统一色板。
"""

from __future__ import annotations

from pygments.style import Style
from pygments.token import (
    Comment,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)
from textual.theme import Theme

# 色值直接取自 theme.py 的变量定义（单一事实来源，防止两处漂移）。
from limbo.ui.theme import LIMBO_DARK, LIMBO_LIGHT

_D = LIMBO_DARK.variables
_L = LIMBO_LIGHT.variables


class LimboDarkStyle(Style):
    """limbo-dark 语法高亮（on bg-elevated #232637）。"""

    name = "limbo-dark"
    background_color = "#232637"
    default_style = ""

    styles = {
        Text: "#DDE1EC",
        Comment: "italic #7F85A0",
        Keyword: "#6B9BD2",
        Keyword.Type: "#6B9BD2",
        Name.Function: "#E8C85A",
        Name.Class: "#6B9BD2",
        Name.Builtin: "#6B9BD2",
        Name.Variable: "#DDE1EC",
        Name.Decorator: "#E8C85A",
        String: "#7EC88F",
        Number: "#E5A04F",
        Operator: "#DDE1EC",
        Punctuation: "#9AA0B5",
        Generic.Heading: "bold #E8C85A",
        Generic.Subheading: "bold #6B9BD2",
        # diff 行：独立前景+底色（F1 修复：原渲染在 syntax 底上对比度 4.36 不达标）
        Generic.Inserted: f"{_D['diff-added']} bg:{_D['diff-added-bg']}",
        Generic.Deleted: f"{_D['diff-removed']} bg:{_D['diff-removed-bg']}",
        Generic.Error: "#F06661",
    }


class LimboLightStyle(Style):
    """limbo-light 语法高亮（on bg-elevated #E5E8F2）。"""

    name = "limbo-light"
    background_color = "#E5E8F2"
    default_style = ""

    styles = {
        Text: "#1E2433",
        Comment: "italic #6B7186",
        Keyword: "#2F5D8F",
        Keyword.Type: "#2F5D8F",
        Name.Function: "#7A5E0C",
        Name.Class: "#2F5D8F",
        Name.Builtin: "#2F5D8F",
        Name.Variable: "#1E2433",
        Name.Decorator: "#7A5E0C",
        String: "#2E7D4C",
        Number: "#9A5B00",
        Operator: "#1E2433",
        Punctuation: "#4B5265",
        Generic.Heading: "bold #7A5E0C",
        Generic.Subheading: "bold #2F5D8F",
        Generic.Inserted: f"{_L['diff-added']} bg:{_L['diff-added-bg']}",
        Generic.Deleted: f"{_L['diff-removed']} bg:{_L['diff-removed-bg']}",
        Generic.Error: "#A93230",
    }


_STYLE_CACHE: dict[tuple[str, ...], type[Style]] = {}


def style_for_theme(theme: Theme) -> type[Style]:
    """按当前主题的 token 现生成 Pygments 样式（limbo-dusk、Omarchy 等）。

    limbo-dark / limbo-light 保留手调的专用样式；其余主题从语义 token 派生，
    同一组颜色只生成一次。
    """
    if theme.name == LIMBO_DARK.name:
        return LimboDarkStyle
    if theme.name == LIMBO_LIGHT.name:
        return LimboLightStyle
    v = theme.variables
    fg = str(theme.foreground)
    primary = str(theme.primary)
    accent = str(theme.accent or theme.primary)
    tertiary = v.get("text-tertiary", fg)
    secondary = v.get("text-secondary", fg)
    success = str(theme.success or fg)
    warning = str(theme.warning or fg)
    error = str(theme.error or fg)
    key = (
        theme.name, fg, primary, accent, tertiary, secondary, success, warning, error,
        v.get("diff-added", success), v.get("diff-added-bg", ""),
        v.get("diff-removed", error), v.get("diff-removed-bg", ""),
        str(theme.panel or theme.background),
    )
    cached = _STYLE_CACHE.get(key)
    if cached is not None:
        return cached
    added_bg = f" bg:{key[10]}" if key[10] else ""
    removed_bg = f" bg:{key[12]}" if key[12] else ""
    style = type(
        "LimboThemeStyle",
        (Style,),
        {
            "name": theme.name,
            "background_color": key[13],
            "default_style": "",
            "styles": {
                Text: fg,
                Comment: f"italic {tertiary}",
                Keyword: primary,
                Keyword.Type: primary,
                Name.Function: accent,
                Name.Class: primary,
                Name.Builtin: primary,
                Name.Variable: fg,
                Name.Decorator: accent,
                String: success,
                Number: warning,
                Operator: fg,
                Punctuation: secondary,
                Generic.Heading: f"bold {accent}",
                Generic.Subheading: f"bold {primary}",
                Generic.Inserted: f"{key[9]}{added_bg}",
                Generic.Deleted: f"{key[11]}{removed_bg}",
                Generic.Error: error,
            },
        },
    )
    _STYLE_CACHE[key] = style
    return style
