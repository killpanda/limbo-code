"""Daily mode (non-coding posture): user-facing notices shared by frontends.

The mechanics live where they belong: the tools in ``ToolRegistry.daily_mode``,
the prompt in ``limbo.prompt``, the toggle in ``Agent.set_daily_mode`` and
persistence in ``SessionMeta.daily_mode``.
"""

from __future__ import annotations

DAILY_USAGE = "用法：/daily-mode on|off（无参则切换）"

# Chat divider labels for a manual switch (TUI, Limbo Dawn).
DAILY_DIVIDER_ON = "☀ Daily mode · 通用助理口吻 · web_fetch 和 memory 已就绪"
DAILY_DIVIDER_OFF = "⌨ 回到 coding · 网页和记忆工具已收起"


def daily_toggle_notice(enabled: bool) -> str:
    if enabled:
        return (
            "Daily mode 已开启：通用助理口吻，新增 web_fetch（读网页）和 memory（长期记忆）；"
            "/memory 查看记忆"
        )
    return "Daily mode 已关闭：回到 coding 模式"


def daily_unchanged_notice(enabled: bool) -> str:
    return f"Daily mode 已是{'开启' if enabled else '关闭'}状态"
