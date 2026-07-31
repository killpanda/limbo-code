"""Tests for the goal-mode pure functions and the verify executor (LIM-40)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from limbo.goal import (
    STATUS_ACTIVE,
    STATUS_EXHAUSTED,
    STATUS_PASSED,
    GoalClear,
    GoalQuery,
    GoalSet,
    GoalState,
    GoalVerify,
    VerifyResult,
    build_goal_prompt,
    build_initial_prompt,
    build_wrapup_prompt,
    next_state,
    parse_goal_args,
    run_verify,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


# -- parse_goal_args ----------------------------------------------------------


def test_parse_empty_is_query():
    assert isinstance(parse_goal_args(""), GoalQuery)
    assert isinstance(parse_goal_args("   "), GoalQuery)


def test_parse_clear():
    assert isinstance(parse_goal_args("clear"), GoalClear)


def test_parse_verify():
    cmd = parse_goal_args("verify uv run python -m pytest -x")
    assert isinstance(cmd, GoalVerify)
    assert cmd.command == "uv run python -m pytest -x"


def test_parse_verify_without_command():
    cmd = parse_goal_args("verify")
    assert isinstance(cmd, GoalVerify)
    assert cmd.command == ""


def test_parse_goal_text():
    cmd = parse_goal_args("为 tools/ 补齐测试\n验收标准：全绿")
    assert isinstance(cmd, GoalSet)
    assert "补齐测试" in cmd.text


# -- prompt templates ---------------------------------------------------------


def test_initial_prompt_with_verify():
    state = GoalState(text="补测试", verify_command="pytest -x")
    prompt = build_initial_prompt(state)
    assert "补测试" in prompt
    assert "pytest -x" in prompt
    assert "退出码 0" in prompt


def test_initial_prompt_without_verify_notes_no_auto_loop():
    state = GoalState(text="补测试")
    prompt = build_initial_prompt(state)
    assert "尚未设置验收命令" in prompt


def test_goal_prompt_feeds_failure_back_verbatim():
    failure = "FAILED tests/tools/test_x.py::test_y - AssertionError: boom"
    state = GoalState(
        text="补测试", verify_command="pytest -x", rounds_completed=2
    )
    result = VerifyResult(exit_code=1, output_tail=failure)
    prompt = build_goal_prompt(state, result)
    # Failure output must be fed back verbatim (RFC: 一字不改地交回去).
    assert failure in prompt
    assert "退出码 1" in prompt
    assert "第 3 轮验收未通过" in prompt
    assert "歧义" in prompt


def test_goal_prompt_timeout_variant():
    state = GoalState(text="g", verify_command="make test")
    result = VerifyResult(exit_code=None, timed_out=True)
    assert "超时" in build_goal_prompt(state, result)


def test_wrapup_prompt_forbids_tools_and_guides_recovery():
    state = GoalState(text="补测试", max_rounds=10)
    prompt = build_wrapup_prompt(state)
    assert "10 轮" in prompt
    assert "请不要使用任何工具" in prompt
    assert "继续" in prompt  # LIM-35 recovery guidance
    assert "/goal clear" in prompt


# -- next_state ---------------------------------------------------------------


def test_next_state_pass():
    state = GoalState(text="g")
    new = next_state(state, VerifyResult(exit_code=0))
    assert new.status == STATUS_PASSED
    assert new.rounds_completed == 1


def test_next_state_failure_stays_active_within_budget():
    state = GoalState(text="g", max_rounds=10)
    new = next_state(state, VerifyResult(exit_code=1))
    assert new.status == STATUS_ACTIVE
    assert new.rounds_completed == 1


def test_next_state_exhausted_at_budget():
    state = GoalState(text="g", max_rounds=2, rounds_completed=1)
    new = next_state(state, VerifyResult(exit_code=1))
    assert new.status == STATUS_EXHAUSTED
    assert new.rounds_completed == 2


# -- run_verify ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_verify_exit_zero(workdir):
    result = await run_verify("exit 0", workdir)
    assert result.exit_code == 0
    assert not (result.timed_out or result.cancelled or result.refused)


@pytest.mark.asyncio
async def test_run_verify_nonzero_and_output_tail(workdir):
    result = await run_verify("echo some-failure-output && exit 3", workdir)
    assert result.exit_code == 3
    assert "some-failure-output" in result.output_tail


@pytest.mark.asyncio
async def test_run_verify_truncates_to_tail(workdir):
    # 3000 numbered lines: the tail (not the head) must survive.
    result = await run_verify(
        "seq 1 3000; exit 1", workdir
    )
    assert result.exit_code == 1
    assert "3000" in result.output_tail
    assert "\n1\n" not in f"\n{result.output_tail}"


@pytest.mark.asyncio
async def test_run_verify_timeout_kills_process(workdir):
    sleep = "timeout" if sys.platform == "win32" else "sleep"
    result = await run_verify(f"{sleep} 30", workdir, timeout_ms=300)
    assert result.exit_code is None
    assert result.timed_out


@pytest.mark.asyncio
async def test_run_verify_dangerous_refused(workdir):
    result = await run_verify("rm -rf /", workdir, dangerous_patterns=["rm"])
    assert result.refused
    assert result.exit_code is None


@pytest.mark.asyncio
async def test_run_verify_dangerous_default_fallback(workdir):
    # No explicit patterns: falls back to DEFAULT_DANGEROUS_COMMANDS (the
    # same posture as the bash tool), which includes "rm".
    result = await run_verify("rm -rf /", workdir)
    assert result.refused
    result = await run_verify("git reset --hard", workdir)
    assert result.refused


@pytest.mark.asyncio
async def test_run_verify_cancellation_kills_process(workdir):
    sleep = "timeout" if sys.platform == "win32" else "sleep"
    task = asyncio.create_task(run_verify(f"{sleep} 30", workdir))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
