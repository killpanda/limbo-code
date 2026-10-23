"""User-side verbs shared by the TUI and headless frontends (design §7.3).

- ``/task [<id>]`` · ``/task <action> <id|all> [text]`` — the board and
  direct control: pause, resume, cancel, retry, approve, reject, merge,
  prio, edit, send.
- ``/agents scan`` · ``/agents init`` — project facts and the suggested
  project profiles (``init`` writes them).
- ``@name text`` — talk to one subagent directly.

Each returns the text to show the user. Non-UI layer.
"""

from __future__ import annotations

import re

from limbo.subagents.profiles import PROJECT_AGENTS_DIR
from limbo.subagents.project_scan import format_facts, suggest_profiles
from limbo.subagents.supervisor import STATUS_LABEL, Supervisor

TASK_USAGE = (
    "用法：/task 查看任务板 · /task <id> 查看详情 · "
    "/task <pause|resume|cancel|retry|approve|reject|merge> <id|all> [说明] · "
    "/task prio <id> <n> · /task edit <id> <新说明> · /task send <id> <消息>；"
    "也可以直接 @子代理名 <消息>"
)

_ACTIONS = ("pause", "resume", "cancel", "retry", "approve", "reject", "merge")

_MENTION = re.compile(r"^@([\w.-]+)\s+(.+)$", re.DOTALL)


def task_command(sup: Supervisor, arg: str) -> str:
    parts = arg.split(maxsplit=2)
    if not parts:
        return sup.status_text() + "\n\n" + TASK_USAGE if sup.board.tasks else TASK_USAGE
    action = parts[0].lower()
    if action in ("help", "-h", "--help"):
        return TASK_USAGE
    if len(parts) == 1:
        return sup.status_text(parts[0])
    ref = parts[1]
    rest = parts[2] if len(parts) > 2 else ""
    if action in _ACTIONS:
        fresh = action == "retry" and rest.strip() == "fresh"
        return sup.control(ref, action, note="" if fresh else rest, fresh=fresh, by="user")
    if action in ("prio", "priority", "reprioritize"):
        try:
            priority = int(rest.strip())
        except ValueError:
            return "用法：/task prio <id> <整数，越大越先>"
        return sup.control(ref, "reprioritize", priority=priority, by="user")
    if action == "edit":
        if not rest.strip():
            return "用法：/task edit <id> <新的任务说明>"
        return sup.control(ref, "edit", brief=rest, by="user")
    if action == "send":
        return sup.send(ref, rest, by="user")
    return f"未知动作 {action}\n{TASK_USAGE}"


def direct_message(sup: Supervisor, text: str) -> str | None:
    """``@name text`` → the reply line; None when it names no subagent
    (the text is then an ordinary message)."""
    match = _MENTION.match(text.strip())
    if match is None:
        return None
    if sup.board.find(match.group(1)) is None:
        return None
    return sup.send(match.group(1), match.group(2).strip(), by="user")


def agents_command(sup: Supervisor, arg: str) -> str:
    action = arg.strip().lower()
    if action == "scan":
        lines = [format_facts(sup.facts), "", "可用的子代理角色："]
        for p in sup.profiles.values():
            source = "内置" if p.builtin else str(p.path or "")
            lines.append(f"- {p.name}（{p.tools}）{p.description} · {source}")
        suggestions = suggest_profiles(sup.facts, sup.workdir)
        if suggestions:
            lines.append("")
            lines.append("建议的项目角色（/agents init 写入 .agents/agents/）：")
            lines += [f"- {s.name}：{s.reason}" for s in suggestions]
        return "\n".join(lines)
    if action == "init":
        suggestions = suggest_profiles(sup.facts, sup.workdir)
        if not suggestions:
            return f"没有新的建议角色（{PROJECT_AGENTS_DIR}/ 已覆盖，或项目没有可切分的子包）"
        written = []
        for s in suggestions:
            s.path.parent.mkdir(parents=True, exist_ok=True)
            s.path.write_text(s.content, encoding="utf-8")
            written.append(f"- {s.path.relative_to(sup.workdir)}（{s.reason}）")
        return "已写入项目角色，下次会话生效：\n" + "\n".join(written)
    if action in ("", "list", "status"):
        return sup.status_text()
    return "用法：/agents（面板）· /agents scan（项目画像与角色）· /agents init（生成项目角色）"


def counts_label(sup: Supervisor) -> str:
    """``2 运行中 · 1 等待中`` for the status pill; empty when idle."""
    counts = sup.summary_counts()
    return " · ".join(f"{n} {STATUS_LABEL.get(status, status)}" for status, n in counts.items())
