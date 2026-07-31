"""Closed-loop goal mode (/goal): state, prompt templates, verify executor.

Non-UI layer (RFC LIM-40 v1.2): this module must never import textual.
The loop's gate is the verify command's exit code — 0 means the goal is
met; anything else means the failure output is fed back verbatim into the
next round. See docs in goal_driver.py for the orchestration layer.
"""

from __future__ import annotations

import asyncio
import os
import re
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
class GoalClear:
    pass


@dataclass(frozen=True)
class GoalQuery:
    pass


GoalCommand = GoalSet | GoalClear | GoalQuery


def parse_goal_args(arg: str) -> GoalCommand:
    """Parse the /goal argument into a command variant.

    Only two commands exist (D5 revised): ``/goal <目标>`` and
    ``/goal clear``; anything that isn't ``clear`` (or empty) is goal text.
    """
    arg = arg.strip()
    if not arg:
        return GoalQuery()
    if arg == "clear":
        return GoalClear()
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


# -- prompt templates ----------------------------------------------------------

VERIFY_PROPOSAL_TAG = "verify_proposal"
_MAX_PROPOSALS = 3


def build_proposer_prompt(state: GoalState) -> str:
    """First (proposal) round after ``/goal <text>`` (M2): the model explores
    the repo and proposes the acceptance command(s); it must stop after the
    proposal and wait for the user's confirmation — no real work yet."""
    return (
        "🎯 Goal 模式已启动\n\n"
        f"目标：{state.text}\n\n"
        "你的第一个任务（先不要动手修改任何代码）：探索当前仓库，找出能客观判定"
        "该目标是否完成的验收命令（例如跑哪些测试、lint、typecheck）。要求：\n"
        "- 命令必须能在当前工作目录直接用 bash 运行，退出码 0 = 通过、非 0 = 未通过；\n"
        "- 优先复用仓库已有的测试/检查入口（Makefile、pyproject、package.json 等）；\n"
        f"- 给出 1–{_MAX_PROPOSALS} 条候选（可以是一条组合命令）；\n"
        "- 若该目标没有客观可判定的验收方式（例如纯审美类目标），如实说明，不要硬造。\n"
        "探索完成后，用以下格式输出提议，然后停止，等我确认：\n"
        f"<{VERIFY_PROPOSAL_TAG}>\n"
        "<command>命令1</command>\n"
        "<command>命令2（可选）</command>\n"
        "<rationale>一句话说明理由</rationale>\n"
        f"</{VERIFY_PROPOSAL_TAG}>\n"
        f"若没有客观验收方式，输出 <{VERIFY_PROPOSAL_TAG}><none/></{VERIFY_PROPOSAL_TAG}> 并解释。"
    )


def parse_verify_proposal(text: str) -> list[str] | None:
    """Parse the model's verify proposal out of its response text.

    Returns ``None`` when no ``<verify_proposal>`` block is present, ``[]``
    for an explicit ``<none/>`` (no objective gate exists), otherwise up to
    3 candidate commands (stripped, de-duplicated, order preserved).
    """
    blocks = re.findall(
        rf"<{VERIFY_PROPOSAL_TAG}>(.*?)</{VERIFY_PROPOSAL_TAG}>",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        return None
    block = blocks[-1]  # the final block wins if the model drafted several
    if "<none/>" in block or "<none>" in block:
        return []
    commands: list[str] = []
    for match in re.findall(r"<command>(.*?)</command>", block, flags=re.DOTALL):
        command = " ".join(match.split())
        if command and command not in commands:
            commands.append(command)
    return commands[:_MAX_PROPOSALS]


def build_initial_prompt(state: GoalState) -> str:
    """First-round instruction after ``/goal <text>`` (goal block pinned)."""
    if state.verify_command:
        gate = f"验收标准将由命令 `{state.verify_command}` 判定（退出码 0 = 通过）。"
    else:
        gate = "尚未设置验收命令；本轮结束后不会自动验收续轮。"
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
