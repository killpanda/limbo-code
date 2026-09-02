"""WCAG 对比度校验（RFC LIM-16 v1.1 P2-3 / Test Plan §7.5）。

全枚举「前景 token × 三级背景（bg-base / bg-surface / bg-elevated）」矩阵，
外加错误底/diff 底上的专用组合。规则分级：

- **在用（used）文字组合**（§6.3 语义用色规范中实际使用的配对）：必须 ≥ 4.5:1，否则 FAIL
- **未在用组合**：≥ 4.5 记 OK，3.0~4.5 记 WARN（可用但仅限装饰/符号），< 3.0 记 BAD
- 任何 BAD 或在用组合 FAIL 都会让 ``check_palettes`` 返回非零

色值直接读 ``limbo.ui.theme`` 的主题定义，单一事实来源，不会漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from limbo.ui.theme import LIMBO_DARK, LIMBO_LIGHT, palette_snapshot

TEXT_MIN = 4.5
DECORATIVE_MIN = 3.0

_BACKGROUNDS = ("bg-base", "bg-surface", "bg-elevated")
# 前景阶与语义色都会作为"文字"出现在三级背景上，全部纳入枚举。
_FOREGROUNDS = (
    "text-primary",
    "text-secondary",
    "text-tertiary",
    "primary",
    "accent",
    "success",
    "warning",
    "error",
    "diff-added",
    "diff-removed",
    "user-message-fg",
)

# §6.3 规范中实际使用的「文字 on 背景」组合（低于 4.5 即 FAIL）。
# 其余组合只报告不拦截，覆盖 text-tertiary on bg-elevated 这类边界情况。
USED_PAIRS = frozenset(
    {
        ("text-primary", "bg-base"),
        ("text-primary", "bg-surface"),
        ("text-secondary", "bg-base"),
        ("text-secondary", "bg-surface"),
        ("text-tertiary", "bg-base"),  # thinking 推理文本、quote/hr
        ("primary", "bg-base"),  # 状态栏 thinking/tool、prompt 符
        ("primary", "bg-surface"),  # 用户消息左竖条、工具卡 running 头
        ("primary", "bg-elevated"),  # 代码块左边框
        ("accent", "bg-base"),  # Markdown 标题/inline code 提示
        ("accent", "bg-elevated"),  # 菜单/picker 选中态文字
        ("success", "bg-surface"),  # 工具卡 ✓
        ("warning", "bg-base"),  # 未来的"等待确认"
        ("error", "bg-surface"),  # diff 删除前景 on 卡片底（历史输出无 diff 底时）
        ("error-bright", "error-bg"),  # 错误块/失败卡片文字
        # F1：edit diff 的真实渲染路径（pygments Inserted/Deleted on diff 底）
        ("diff-added", "diff-added-bg"),
        ("diff-removed", "diff-removed-bg"),
        # 用户消息蓝染底上的文字（新语义色，与 agent 中性前景区分）
        ("user-message-fg", "user-message-bg"),
    }
)

# 额外检查的非三级背景组合（报告 + 按 used 判定）。
_EXTRA_PAIRS = (
    ("error-bright", "error-bg"),
    ("diff-added", "diff-added-bg"),
    ("diff-removed", "diff-removed-bg"),
    ("user-message-fg", "user-message-bg"),  # 用户消息蓝染底文字
)


def relative_luminance(hex_color: str) -> float:
    """WCAG 相对亮度（sRGB，6 位 hex）。"""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def linearize(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def contrast_ratio(fg: str, bg: str) -> float:
    l1, l2 = relative_luminance(fg), relative_luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


@dataclass(frozen=True)
class CheckResult:
    theme: str
    fg: str
    bg: str
    ratio: float
    used: bool
    status: str  # "OK" | "WARN" | "FAIL" | "BAD"


def check_theme(theme_name: str, palette: dict[str, str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    pairs: Iterable[tuple[str, str]] = [
        (fg, bg) for fg in _FOREGROUNDS for bg in _BACKGROUNDS
    ]
    for fg, bg in list(pairs) + list(_EXTRA_PAIRS):
        if fg not in palette or bg not in palette:
            continue
        ratio = contrast_ratio(palette[fg], palette[bg])
        used = (fg, bg) in USED_PAIRS
        if used and ratio < TEXT_MIN:
            status = "FAIL"
        elif ratio >= TEXT_MIN:
            status = "OK"
        elif ratio >= DECORATIVE_MIN:
            status = "WARN"
        else:
            status = "BAD"
        results.append(CheckResult(theme_name, fg, bg, ratio, used, status))
    return results


def check_palettes() -> tuple[list[CheckResult], int]:
    """校验 limbo-dark / limbo-light。返回 (结果列表, 失败数)。"""
    results: list[CheckResult] = []
    for theme in (LIMBO_DARK, LIMBO_LIGHT):
        results.extend(check_theme(theme.name, palette_snapshot(theme)))
    failures = sum(1 for r in results if r.status in ("FAIL", "BAD"))
    return results, failures


def format_report(results: list[CheckResult]) -> str:
    lines = []
    current = None
    for r in results:
        if r.theme != current:
            current = r.theme
            lines.append(f"\n== {current} ==")
            lines.append(f"{'foreground':<16} {'background':<16} {'ratio':>6}  {'used':<4} status")
        mark = "yes" if r.used else "-"
        lines.append(
            f"{r.fg:<16} {r.bg:<16} {r.ratio:>5.2f}:1 {mark:<4} {r.status}"
        )
    return "\n".join(lines)
