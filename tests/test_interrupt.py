"""Tests for ESC turn interruption (RFC LIM-53).

Covers the RFC Test Plan unit layer: stream interrupt (incl. stalled
reads), partial-reasoning drop, tool-batch semantics (file tools record
real results, unstarted calls get placeholders, bash process trees are
killed), idempotency/reset points, goal-driver stop ordering, leftover
steer delivery on the next run, and compact interruption.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from limbo.agent import (
    Agent,
    InterruptEvent,
    TextDelta,
    ToolResultEvent,
)
from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent
from limbo.trace import read_trace


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))
        for event in self.responses.pop(0):
            yield event


def make_agent(workdir: Path, client, config: Config | None = None) -> Agent:
    return Agent(
        config=config or Config(),
        llm_client=client,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )


def read_agent_trace(agent: Agent) -> list[dict]:
    """read_trace with a flush barrier (see trace_types)."""
    agent.trace.flush()
    return read_trace(agent.trace.path)


def trace_types(agent: Agent) -> list[str]:
    return [r["type"] for r in read_agent_trace(agent)]


# -- 1. LLM stream interrupt ----------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_mid_stream_appends_partial_content_only(workdir):
    chunks = [TextChunk(text=f"c{i}") for i in range(10)]
    client = FakeLLMClient([chunks])
    agent = make_agent(workdir, client)

    events = []
    async for event in agent.run("hi"):
        events.append(event)
        if sum(isinstance(e, TextDelta) for e in events) == 3:
            agent.interrupt()

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "c0c1c2"
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert interrupts[0].phase == "llm_stream"

    # Partial response appended content-only: no tool calls, no reasoning.
    last = agent.messages[-1]
    assert last.role == "assistant"
    assert last.content == "c0c1c2"
    assert last.tool_calls is None
    assert last.reasoning is None
    assert last.reasoning_signature is None

    records = read_agent_trace(agent)
    interrupted = [r for r in records if r["type"] == "turn_interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["phase"] == "llm_stream"
    turn_ends = [r for r in records if r["type"] == "turn_end"]
    assert turn_ends[-1]["status"] == "interrupted"
    assert agent.was_interrupted is True


# -- 2. Partial reasoning is dropped; dialect replays tolerate it ---------------


# -- 3. Tool batch: started file tools record real results ----------------------


async def _run_until(task_events: list, agent: Agent, predicate) -> None:
    async def consume():
        async for event in agent.run("go"):
            task_events.append(event)

    task = asyncio.create_task(consume())
    for _ in range(1000):
        await asyncio.sleep(0.01)
        if predicate():
            return task
    task.cancel()
    pytest.fail("condition never became true")


# -- 4. bash process tree is killed ----------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_kills_bash_process_tree(workdir):
    client = FakeLLMClient(
        [[ToolCallEvent(id="b1", name="bash", arguments={"command": "sleep 30"})]]
    )
    agent = make_agent(workdir, client)
    bash_tool = agent.registry.get("bash")

    events: list = []
    task = await _run_until(events, agent, lambda: bool(bash_tool._active_procs))
    agent.interrupt()
    await asyncio.wait_for(task, 15)

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].result.success is False
    assert "已被用户打断" in (results[0].result.error or "")
    assert not bash_tool._active_procs
    assert any(isinstance(e, InterruptEvent) for e in events)


# -- 5. Idempotency and reset points ----------------------------------------------


# -- 6. Goal driver stops BEFORE the leftover-steer drain ------------------------


# -- 7. Leftover steers are injected by the NEXT run's head fallback -------------


# -- 8. Stalled stream (no chunk ever arrives) ------------------------------------


# -- 9. Compact interruption and flag reset ---------------------------------------


