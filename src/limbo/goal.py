"""Closed-loop goal mode (/goal): state, prompt templates, verify executor.

Non-UI layer (RFC LIM-40 v1.2): this module must never import textual.
The loop's gate is the verify command's exit code — 0 means the goal is
met; anything else means the failure output is fed back verbatim into the
next round. See docs in goal_driver.py for the orchestration layer.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from limbo.config import DEFAULT_DANGEROUS_COMMANDS
from limbo.tools.base import truncate_tail
from limbo.tools.bash import is_dangerous

# -- state --------------------------------------------------------------------

# Goal lifecycle: active → passed | exhausted | cleared; exhausted → active
# happens when the user sends any message (budget resets, LIM-35/D3).
STATUS_ACTIVE = "active"
STATUS_PASSED = "passed"
STATUS_EXHAUSTED = "exhausted"
STATUS_CLEARED = "cleared"


class GoalState(BaseModel):
    """One session-scoped goal (persisted on SessionMeta.goal)."""

    text: str
    # Shell command whose exit code gates the loop; empty = no auto-verify.
    verify_command: str = ""
    rounds_completed: int = 0
    max_rounds: int = 10
    status: str = STATUS_ACTIVE


# -- /goal argument parsing ---------------------------------------------------


@dataclass(frozen=True)
class GoalSet:
    text: str


@dataclass(frozen=True)
class GoalVerify:
    command: str


@dataclass(frozen=True)
class GoalClear:
    pass


@dataclass(frozen=True)
class GoalQuery:
    pass


GoalCommand = GoalSet | GoalVerify | GoalClear | GoalQuery


def parse_goal_args(arg: str) -> GoalCommand:
    """Parse the /goal argument into a command variant.

    ``/goal`` → query, ``/goal clear`` → clear, ``/goal verify <cmd>`` →
    set the verify command; anything else is the goal text itself (a goal
    whose text literally starts with "verify " needs quoting by wording —
    documented limitation of the single-command free-text form, D5).
    """
    arg = arg.strip()
    if not arg:
        return GoalQuery()
    if arg == "clear":
        return GoalClear()
    if arg == "verify" or arg.startswith("verify "):
        return GoalVerify(arg.removeprefix("verify").strip())
    return GoalSet(arg)


# -- verify execution ---------------------------------------------------------


@dataclass(frozen=True)
class VerifyResult:
    # None when the command never produced an exit code (timeout / cancel /
    # refused by the dangerous-command filter).
    exit_code: int | None
    output_tail: str = ""
    timed_out: bool = False
    cancelled: bool = False
    refused: bool = False


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the verify process's whole group (mirrors bash.py).

    The shell runs in its own session (``start_new_session``), so a group
    kill reaches detached children a plain ``proc.kill()`` would orphan.
    """
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    else:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
    except (asyncio.TimeoutError, ProcessLookupError, OSError):
        pass


async def run_verify(
    command: str,
    workdir: Path,
    *,
    timeout_ms: int = 600_000,
    dangerous_patterns: list[str] | None = None,
) -> VerifyResult:
    """Run the acceptance command; the exit code is the loop's gate.

    stdout+stderr are merged and tail-truncated (2000 lines / 50KB, keep
    the tail — failure stacks live at the end). Cancellation (Esc route via
    GoalDriver.cancel_verify) kills the process tree and re-raises so the
    caller can turn it into a ``cancelled`` result.
    """
    patterns = (
        list(dangerous_patterns)
        if dangerous_patterns is not None
        else list(DEFAULT_DANGEROUS_COMMANDS)
    )
    if is_dangerous(command, patterns):
        return VerifyResult(exit_code=None, refused=True)
    # Same shell dialect as the bash tool: bash -c, own process group.
    proc = await asyncio.create_subprocess_exec(
        shutil.which("bash") or "/bin/bash",
        "-c",
        command,
        cwd=str(workdir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        # Own process group so cancellation/timeout can kill the tree.
        start_new_session=(sys.platform != "win32"),
    )
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_ms / 1000
        )
    except asyncio.TimeoutError:
        await _kill_process_tree(proc)
        return VerifyResult(exit_code=None, timed_out=True)
    except asyncio.CancelledError:
        await _kill_process_tree(proc)
        raise
    text = stdout.decode("utf-8", errors="replace") if stdout else ""
    tail = truncate_tail(text)
    return VerifyResult(exit_code=proc.returncode, output_tail=tail.content)


# -- prompt templates ---------------------------------------------------------


def build_initial_prompt(state: GoalState) -> str:
    """First-round instruction after ``/goal <text>`` (goal block pinned)."""
    if state.verify_command:
        gate = f"验收标准将由命令 `{state.verify_command}` 判定（退出码 0 = 通过）。"
    else:
        gate = (
            "尚未设置验收命令（可用 /goal verify <命令> 设置）；"
            "本轮结束后不会自动验收续轮。"
        )
    return (
        "🎯 Goal 模式已启动\n\n"
        f"目标：{state.text}\n\n"
        f"{gate}\n\n"
        "循环规则：\n"
        "- 请开始朝着目标工作；你认为完成后，我会执行验收命令。\n"
        "- 验收未通过时，失败输出会原样交回给你，继续修复，直到通过。\n"
        "- 若验收标准本身有歧义，先向我提问，不要自行猜测。"
    )


def build_goal_prompt(state: GoalState, result: VerifyResult) -> str:
    """Self-prompt for the next round: failure output fed back verbatim."""
    round_no = state.rounds_completed + 1
    if result.timed_out:
        outcome = f"验收命令 `{state.verify_command}` 执行超时，未完成。"
    else:
        outcome = (
            f"验收命令 `{state.verify_command}` 退出码 {result.exit_code}，"
            f"失败输出如下（原样）：\n<verify_output>\n"
            f"{result.output_tail}\n</verify_output>"
        )
    return (
        f"🎯 Goal（第 {round_no} 轮验收未通过，进入第 {round_no + 1} 轮）\n\n"
        f"目标：{state.text}\n\n"
        f"{outcome}\n\n"
        "循环规则：\n"
        "- 阅读失败输出，定位并修复，然后继续朝验收标准推进。\n"
        "- 验收标准全部满足才算完成；不要凭印象猜测哪里出了问题。\n"
        "- 若验收标准本身有歧义，先向用户提问，不要自行猜测。"
    )


def build_wrapup_prompt(state: GoalState) -> str:
    """Budget-exhausted wrap-up (LIM-37): no tools, summarize + recovery
    guidance (LIM-35). Runs as a normal turn but is not counted as a round
    and is not verified afterwards."""
    return (
        f"🎯 Goal 已达最大轮次（{state.max_rounds} 轮）仍未通过验收。\n\n"
        f"目标：{state.text}\n\n"
        "请不要使用任何工具，直接总结汇报：\n"
        "1. 已完成什么；\n"
        "2. 卡在哪里（最近失败的原因）；\n"
        "3. 建议怎么继续。\n"
        "最后告诉用户：发送任意消息（如“继续”）即可带着新预算继续闭环；"
        "/goal clear 可退出 goal 模式。"
    )


# -- state machine ------------------------------------------------------------


def next_state(state: GoalState, result: VerifyResult) -> GoalState:
    """Advance the goal after one verify (one verify = one round)."""
    rounds = state.rounds_completed + 1
    if result.exit_code == 0:
        return state.model_copy(
            update={"rounds_completed": rounds, "status": STATUS_PASSED}
        )
    status = STATUS_EXHAUSTED if rounds >= state.max_rounds else STATUS_ACTIVE
    return state.model_copy(update={"rounds_completed": rounds, "status": status})
