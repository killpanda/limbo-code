"""Tests for the GoalDriver orchestrator (LIM-40, RFC v1.2).

Real ``Agent`` + scripted fake LLM client drive the turns; ``run_verify``
is monkeypatched to a stub so no real subprocess runs. The FakeAgent at
the bottom covers timing-sensitive steer/E1 paths that a real Agent can
only reach through error-exit races.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

import limbo.goal_driver as goal_driver_module
from limbo.agent import Agent, SteerEvent, TextDelta
from limbo.config import Config
from limbo.goal import STATUS_EXHAUSTED, VerifyResult
from limbo.goal_driver import (
    GoalDriver,
    GoalExhausted,
    GoalPassed,
    GoalResumed,
    GoalVerifyResultEvent,
    GoalVerifyStarted,
)
from limbo.models import TextChunk
from limbo.sessions import SessionMeta
from limbo.trace import TraceLogger, read_trace


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append([m for m in messages])
        for event in self.responses.pop(0):
            yield event


def make_agent(workdir: Path, responses) -> Agent:
    return Agent(
        config=Config(),
        llm_client=FakeLLMClient(responses),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )


def make_driver(
    workdir: Path, responses, *, max_rounds: int = 10
) -> GoalDriver:
    agent = make_agent(workdir, responses)
    return GoalDriver(agent, workdir, max_rounds=max_rounds)


def stub_verify(monkeypatch, results):
    """Stub run_verify with a scripted sequence of VerifyResults."""
    queue = list(results)
    seen = []

    async def fake(command, workdir, **kwargs):
        seen.append(command)
        return queue.pop(0)

    monkeypatch.setattr(goal_driver_module, "run_verify", fake)
    return seen


async def collect(run):
    return [event async for event in run]


def text_of(events) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


# -- basic loop ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_continues_on_failure_and_stops_on_pass(
    workdir, monkeypatch
):
    # Two turns: round 1 "fix attempt", round 2 "fixed". Verify: [1, 0].
    driver = make_driver(
        workdir, [[TextChunk(text="try fix")], [TextChunk(text="done")]]
    )
    commands = stub_verify(
        monkeypatch,
        [
            VerifyResult(exit_code=1, output_tail="FAILED test_x"),
            VerifyResult(exit_code=0),
        ],
    )
    driver.set_goal("补测试", verify_command="pytest -x")

    events = await collect(driver.run("start"))

    assert commands == ["pytest -x", "pytest -x"]
    assert any(isinstance(e, GoalPassed) and e.rounds == 2 for e in events)
    # Second turn's first user message is the self-prompt with the failure
    # output fed back verbatim.
    agent = driver._agent
    second_turn_input = agent.messages[1].content  # round-1 user input
    followups = [
        m.content for m in agent.messages if m.role == "user" and "FAILED test_x" in str(m.content)
    ]
    assert followups, "expected the verbatim failure output in a user message"
    assert second_turn_input == "start"
    state = driver.status()
    assert state is not None and state.status == "passed"


@pytest.mark.asyncio
async def test_no_verify_command_single_turn(workdir, monkeypatch):
    driver = make_driver(workdir, [[TextChunk(text="ok")]])
    commands = stub_verify(monkeypatch, [])
    driver.set_goal("补测试")  # no verify command yet

    events = await collect(driver.run("start"))

    assert commands == []
    assert not any(isinstance(e, GoalVerifyStarted) for e in events)
    assert text_of(events) == "ok"


@pytest.mark.asyncio
async def test_exhausted_wrapup_and_resume_resets_budget(workdir, monkeypatch):
    driver = make_driver(
        workdir,
        [
            [TextChunk(text="round1")],
            [TextChunk(text="round2")],
            [TextChunk(text="wrapup summary")],
            [TextChunk(text="resumed round")],
        ],
        max_rounds=2,
    )
    stub_verify(
        monkeypatch,
        [
            VerifyResult(exit_code=1, output_tail="f1"),
            VerifyResult(exit_code=1, output_tail="f2"),
            VerifyResult(exit_code=0),
        ],
    )
    driver.set_goal("g", verify_command="pytest")

    events = await collect(driver.run("start"))

    # Budget exhausted: wrapup turn ran (uncounted), then GoalExhausted.
    assert any(isinstance(e, GoalExhausted) and e.rounds == 2 for e in events)
    assert "wrapup summary" in text_of(events)
    state = driver.status()
    assert state is not None and state.status == STATUS_EXHAUSTED

    # Any user message resumes the loop with a reset budget (D3/LIM-35).
    events = await collect(driver.run("继续"))
    assert any(isinstance(e, GoalResumed) for e in events)
    assert any(isinstance(e, GoalPassed) for e in events)
    state = driver.status()
    assert state is not None and state.rounds_completed == 1


@pytest.mark.asyncio
async def test_clear_stops_continuation(workdir, monkeypatch):
    driver = make_driver(workdir, [[TextChunk(text="round1")]])
    # The stub clears the goal when the verify runs (mid-loop clear).
    async def verify_then_clear(command, workdir, **kwargs):
        driver.clear()
        return VerifyResult(exit_code=1, output_tail="fail")

    monkeypatch.setattr(goal_driver_module, "run_verify", verify_then_clear)
    driver.set_goal("g", verify_command="pytest")

    events = await collect(driver.run("start"))

    # Verify ran, the failure was reported, but no second turn started.
    assert any(isinstance(e, GoalVerifyResultEvent) for e in events)
    assert not any(isinstance(e, GoalPassed) for e in events)
    assert len(driver._agent.messages) <= 4  # system + user + assistant
    assert driver.status() is not None and driver.status().status == "cleared"


@pytest.mark.asyncio
async def test_verify_cancel_stops_loop_keeps_goal(workdir, monkeypatch):
    driver = make_driver(workdir, [[TextChunk(text="round1")]])

    async def cancelled_verify(command, workdir, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(goal_driver_module, "run_verify", cancelled_verify)
    driver.set_goal("g", verify_command="pytest")

    events = await collect(driver.run("start"))

    assert any(
        isinstance(e, GoalVerifyResultEvent) and e.result.cancelled for e in events
    )
    assert driver.status() is not None and driver.status().status == "active"


# -- running flag / single-flight ----------------------------------------------


@pytest.mark.asyncio
async def test_running_flag_eager_and_reset(workdir, monkeypatch):
    driver = make_driver(workdir, [[TextChunk(text="ok")]])
    stub_verify(monkeypatch, [])
    assert not driver.running
    stream = driver.run("start")  # eager: set before the first event
    assert driver.running
    await collect(stream)
    assert not driver.running


@pytest.mark.asyncio
async def test_second_run_rejected_while_running(workdir):
    driver = make_driver(workdir, [[TextChunk(text="ok")]])
    driver.run("first")
    with pytest.raises(RuntimeError):
        driver.run("second")


# -- steer priority + E1 (FakeAgent for timing control) ------------------------


class FakeAgent:
    """Minimal Agent surface used by GoalDriver (run/drain_steer/meta/trace)."""

    def __init__(self, workdir: Path):
        self.inputs: list[str] = []
        self.session_meta = SessionMeta(id="fake")
        self.trace = TraceLogger(workdir / "trace.jsonl")
        from limbo.steer import SteerQueue

        self._queue = SteerQueue()

    async def run(self, user_input, attachments=None):
        self.inputs.append(user_input)
        yield TextDelta(text=f"turn:{user_input[:20]}")

    def drain_steer(self):
        return self._queue.drain()

    def has_pending_steer(self):
        return len(self._queue) > 0

    def steer(self, text):
        return self._queue.offer(text).id


@pytest.mark.asyncio
async def test_leftover_steer_outranks_goal_and_yields_steer_events(
    workdir, monkeypatch
):
    agent = FakeAgent(workdir)
    driver = GoalDriver(agent, workdir)
    driver.set_goal("g", verify_command="pytest")

    # Queue a steer that surfaces as a turn-boundary leftover: patch drain
    # to return it after the first turn.
    from limbo.steer import SteerItem

    leftover = SteerItem(id="steer123", text="用户插入的消息", attachments=[])
    original_drain = agent.drain_steer
    drained_once = {"done": False}

    def fake_drain():
        if not drained_once["done"]:
            drained_once["done"] = True
            return [leftover]
        return original_drain()

    agent.drain_steer = fake_drain

    async def should_not_run(command, workdir, **kwargs):
        raise AssertionError("verify must not run while a user steer is pending")

    monkeypatch.setattr(goal_driver_module, "run_verify", should_not_run)

    events = []
    async for event in driver.run("goal prompt"):
        events.append(event)
        # After the steered turn ran, swap in a passing verify so the loop
        # can end (the steer turn drains via the real queue now).
        if len(agent.inputs) == 2:
            async def passing(command, workdir, **kwargs):
                return VerifyResult(exit_code=0)

            monkeypatch.setattr(goal_driver_module, "run_verify", passing)

    # E1: one SteerEvent per drained item so the UI flips the queued card.
    steer_events = [e for e in events if isinstance(e, SteerEvent)]
    assert [e.id for e in steer_events] == ["steer123"]
    # The steered message became the next turn's input, before any verify.
    assert agent.inputs[1] == "用户插入的消息"


# -- trace ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trace_events_full_lifecycle(workdir, monkeypatch):
    driver = make_driver(
        workdir, [[TextChunk(text="r1")], [TextChunk(text="r2")]]
    )
    stub_verify(
        monkeypatch,
        [VerifyResult(exit_code=1, output_tail="f"), VerifyResult(exit_code=0)],
    )
    driver.set_goal("g", verify_command="pytest")
    await collect(driver.run("start"))

    kinds = [e.get("type") for e in read_trace(driver._agent.trace.path)]
    assert "goal_set" in kinds
    assert "goal_verify" in kinds
    assert "goal_round" in kinds
    assert "goal_complete" in kinds


# -- layering lock: the driver must stay frontend-agnostic ----------------------


def test_goal_driver_imports_without_textual():
    code = (
        "import sys; import limbo.goal_driver, limbo.goal; "
        "bad = [m for m in sys.modules if m.startswith('textual')]; "
        "assert not bad, f'non-UI layer pulled in textual: {bad}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
