"""Tests for the TurnPump orchestrator (LIM-40, RFC v1.2).

Real ``Agent`` + scripted fake LLM client drive the turns; ``run_verify``
is monkeypatched to a stub so no real subprocess runs. The FakeAgent at
the bottom covers timing-sensitive steer/E1 paths that a real Agent can
only reach through error-exit races.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import limbo.pump as pump_module
from limbo.agent import Agent, CompactionEvent, SteerEvent, TextDelta
from limbo.config import Config
from limbo.goal import VerifyResult
from limbo.models import TextChunk
from limbo.pump import (
    GoalPassed,
    TurnPump,
)
from limbo.sessions import SessionMeta
from limbo.trace import TraceLogger


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


def make_pump(
    workdir: Path, responses, *, max_rounds: int = 10
) -> TurnPump:
    agent = make_agent(workdir, responses)
    return TurnPump(agent, workdir, max_rounds=max_rounds)


def stub_verify(monkeypatch, results):
    """Stub run_verify with a scripted sequence of VerifyResults."""
    queue = list(results)
    seen = []

    async def fake(command, workdir, **kwargs):
        seen.append(command)
        return queue.pop(0)

    monkeypatch.setattr(pump_module, "run_verify", fake)
    return seen


async def collect(run):
    return [event async for event in run]


# -- basic loop ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_continues_on_failure_and_stops_on_pass(
    workdir, monkeypatch
):
    # Two turns: round 1 "fix attempt", round 2 "fixed". Verify: [1, 0].
    driver = make_pump(
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


# -- running flag / single-flight ----------------------------------------------


# -- steer priority + E1 (FakeAgent for timing control) ------------------------


class FakeAgent:
    """Minimal Agent surface used by TurnPump (run/compact/drain/meta/trace)."""

    def __init__(self, workdir: Path):
        self.inputs: list[str] = []
        self.attachments: list[list | None] = []
        self.session_meta = SessionMeta(id="fake")
        self.trace = TraceLogger(workdir / "trace.jsonl")
        self.was_interrupted = False
        # Turns (by input) that end on an LLM error.
        self.failing_inputs: set[str] = set()
        self.last_run_failed = False
        from limbo.steer import SteerQueue

        self._queue = SteerQueue()

    async def run(self, user_input, attachments=None):
        self.inputs.append(user_input)
        self.attachments.append(attachments)
        self.last_run_failed = user_input in self.failing_inputs
        yield TextDelta(text=f"turn:{user_input[:20]}")

    async def compact(self, trigger="manual"):
        yield CompactionEvent(compacted=False, trigger=trigger, reason="fake")

    def drain_steer(self, modes=("steer", "follow_up")):
        return self._queue.drain(modes)

    def has_pending_steer(self):
        return len(self._queue) > 0

    def steer(self, text, attachments=None, mode="steer"):
        return self._queue.offer(text, attachments, mode).id

    def cancel_steer(self, item_id):
        return self._queue.cancel(item_id) is not None

    @property
    def queued_items(self):
        return list(self._queue)

    def persist_meta(self):
        pass


@pytest.mark.asyncio
async def test_leftover_steer_outranks_goal_and_yields_steer_events(
    workdir, monkeypatch
):
    agent = FakeAgent(workdir)
    driver = TurnPump(agent, workdir)
    driver.set_goal("g", verify_command="pytest")

    # Queue a steer that surfaces as a turn-boundary leftover: patch drain
    # to return it after the first turn.
    from limbo.models import Attachment
    from limbo.steer import SteerItem

    image = Attachment(kind="image", path="/tmp/shot.png", name="shot.png")
    leftover = SteerItem(id="steer123", text="用户插入的消息", attachments=[image])
    original_drain = agent.drain_steer
    drained_once = {"done": False}

    def fake_drain(modes=("steer", "follow_up")):
        if not drained_once["done"]:
            drained_once["done"] = True
            return [leftover]
        return original_drain(modes)

    agent.drain_steer = fake_drain

    async def should_not_run(command, workdir, **kwargs):
        raise AssertionError("verify must not run while a user steer is pending")

    monkeypatch.setattr(pump_module, "run_verify", should_not_run)

    events = []
    async for event in driver.run("goal prompt"):
        events.append(event)
        # After the steered turn ran, swap in a passing verify so the loop
        # can end (the steer turn drains via the real queue now).
        if len(agent.inputs) == 2:
            async def passing(command, workdir, **kwargs):
                return VerifyResult(exit_code=0)

            monkeypatch.setattr(pump_module, "run_verify", passing)

    # E1: one SteerEvent per drained item so the UI flips the queued card.
    steer_events = [e for e in events if isinstance(e, SteerEvent)]
    assert [e.id for e in steer_events] == ["steer123"]
    # The steered message became the next turn's input, before any verify.
    assert agent.inputs[1] == "用户插入的消息"
    # Its attachments ride along (the round-boundary drain used to drop them).
    assert agent.attachments[1] == [image]


# -- compact: same single-flight flag -------------------------------------------


# -- trace ---------------------------------------------------------------------


# -- layering lock: the pump must stay frontend-agnostic ------------------------


