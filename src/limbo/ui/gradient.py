"""颜色插值与渐变文字（Limbo Dusk 的字标、进度条、主按钮）。

只用于短文本：逐字素上色会为每个字符生成一个 Rich Style，正文和 diff 从不走
这里。结果按（文本, 颜色, 粗体）缓存，主题切换时颜色变了缓存键自然不同。
"""

from __future__ import annotations

from functools import lru_cache

from rich.style import Style
from rich.text import Text
from textual.theme import Theme

RGB = tuple[int, int, int]


def hex_to_rgb(color: str) -> RGB:
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(rgb: RGB) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(max(0, min(255, round(c))) for c in rgb))


def mix(a: str, b: str, amount: float) -> str:
    """把 ``a`` 向 ``b`` 混合 ``amount``（0 = a，1 = b），RGB 空间线性插值。"""
    amount = max(0.0, min(1.0, amount))
    ra, rb = hex_to_rgb(a), hex_to_rgb(b)
    return rgb_to_hex(tuple(x + (y - x) * amount for x, y in zip(ra, rb)))  # type: ignore[arg-type]


def ramp(stops: tuple[str, ...], t: float) -> str:
    """多段渐变上 ``t``（0..1）处的颜色。"""
    if len(stops) == 1:
        return stops[0]
    t = max(0.0, min(1.0, t))
    pos = t * (len(stops) - 1)
    index = min(int(pos), len(stops) - 2)
    return mix(stops[index], stops[index + 1], pos - index)


def theme_gradient(theme: Theme) -> tuple[str, ...]:
    """主题的渐变色标；非 limbo 主题回落到 primary → accent。"""
    variables = theme.variables
    if "grad-from" in variables and "grad-to" in variables:
        stops = [variables["grad-from"]]
        if "grad-mid" in variables:
            stops.append(variables["grad-mid"])
        stops.append(variables["grad-to"])
        return tuple(stops)
    return (str(theme.primary), str(theme.accent or theme.primary))


@lru_cache(maxsize=256)
def gradient_text(text: str, stops: tuple[str, ...], bold: bool = False) -> Text:
    """按可见字符逐个上色；空格不计入渐变位置，保持字形连贯。"""
    result = Text()
    visible = [i for i, ch in enumerate(text) if not ch.isspace()]
    span = max(1, len(visible) - 1)
    order = {index: n for n, index in enumerate(visible)}
    for i, ch in enumerate(text):
        if i not in order:
            result.append(ch)
            continue
        color = ramp(stops, order[i] / span)
        result.append(ch, Style(color=color, bold=bold))
    return result


@lru_cache(maxsize=256)
def gradient_lines(lines: tuple[str, ...], stops: tuple[str, ...], bold: bool = True) -> Text:
    """多行字标：按列上色，所有行共用同一条水平渐变。"""
    width = max((len(line) for line in lines), default=1)
    span = max(1, width - 1)
    result = Text()
    for row, line in enumerate(lines):
        if row:
            result.append("\n")
        for col, ch in enumerate(line):
            if ch.isspace():
                result.append(ch)
            else:
                result.append(ch, Style(color=ramp(stops, col / span), bold=bold))
    return result
