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
        Generic.Inserted: "#7EC88F",
        Generic.Deleted: "#ED5853",
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
        Generic.Inserted: "#2E7D4C",
        Generic.Deleted: "#C33C38",
        Generic.Error: "#A93230",
    }
