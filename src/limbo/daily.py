"""Daily mode (non-coding posture): user-facing notices shared by frontends.

The mechanics live where they belong: the tools in ``ToolRegistry.daily_mode``,
the prompt in ``limbo.prompt``, the toggle in ``Agent.set_daily_mode`` and
persistence in ``SessionMeta.daily_mode``.
"""

from __future__ import annotations

DAILY_USAGE = "用法：/daily-mode on|off（无参则切换）"


def daily_toggle_notice(enabled: bool) -> str:
    if enabled:
        return (
            "Daily mode 已开启：通用助理口吻，新增读网页、长期记忆、翻历史会话等工具；"
            "/memory 查看记忆，/learn 把经验写进 skill"
        )
    return "Daily mode 已关闭：回到 coding 模式"


def daily_only_notice(command: str) -> str:
    return f"{command} 只在 daily 模式可用（/daily-mode on 开启）"


def daily_unchanged_notice(enabled: bool) -> str:
    return f"Daily mode 已是{'开启' if enabled else '关闭'}状态"
