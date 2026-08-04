"""Tests for ESC turn interruption (RFC LIM-53).

Covers the RFC Test Plan unit layer: stream interrupt (incl. stalled
reads), partial-reasoning drop, tool-batch semantics (file tools record
real results, unstarted calls get placeholders, bash process trees are
killed), idempotency/reset points, goal-driver stop ordering, leftover
steer delivery on the next run, and compact interruption.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

import limbo.goal_driver as goal_driver_module
from limbo.agent import (
    Agent,
    CompactionEvent,
    InterruptEvent,
    SteerEvent,
    TextDelta,
    ToolResultEvent,
)
from limbo.config import Config
from limbo.goal_driver import GoalDriver, GoalVerifyStarted
from limbo.models import TextChunk, ThinkingChunk, ToolCallEvent
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


def trace_types(agent: Agent) -> list[str]:
    return [r["type"] for r in read_trace(agent.trace.path)]


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

    records = read_trace(agent.trace.path)
    interrupted = [r for r in records if r["type"] == "turn_interrupted"]
    assert len(interrupted) == 1
    assert interrupted[0]["phase"] == "llm_stream"
    turn_ends = [r for r in records if r["type"] == "turn_end"]
    assert turn_ends[-1]["status"] == "interrupted"
    assert agent.was_interrupted is True


# -- 2. Partial reasoning is dropped; dialect replays tolerate it ---------------


@pytest.mark.asyncio
async def test_interrupt_drops_partial_reasoning(workdir):
    chunks = [
        ThinkingChunk(text="incomplete thinking", signature="sig"),
        TextChunk(text="c0"),
        TextChunk(text="c1"),
    ]
    client = FakeLLMClient([chunks])
    agent = make_agent(workdir, client)

    events = []
    async for event in agent.run("hi"):
        events.append(event)
        if sum(isinstance(e, TextDelta) for e in events) == 1:
            agent.interrupt()

    last = agent.messages[-1]
    assert last.role == "assistant"
    assert last.reasoning is None
    assert last.reasoning_signature is None

    # Anthropic replay: no thinking block is built for the partial message
    # (a truncated thinking without its full signature would 400).
    from limbo.llm.anthropic_client import _messages_to_anthropic

    _, converted = _messages_to_anthropic(agent.messages)
    assistant = [m for m in converted if m["role"] == "assistant"][-1]
    blocks = assistant["content"]
    assert isinstance(blocks, list)
    assert all(block.get("type") != "thinking" for block in blocks)

    # Kimi-style replay: reasoning_content falls back to "".
    from limbo.llm.openai_client import _messages_to_openai

    converted = _messages_to_openai(agent.messages, include_reasoning=True)
    assistant = [m for m in converted if m["role"] == "assistant"][-1]
    assert assistant["reasoning_content"] == ""


# -- 3. Tool batch: started file tools record real results ----------------------


def _gate_write_tool(agent: Agent, blocked_path: str):
    """Make writes to ``blocked_path`` block on threading events."""
    started = threading.Event()
    release = threading.Event()
    write_tool = agent.registry.get("write")
    original_run = write_tool.run

    def slow_run(arguments):
        if arguments.get("path") == blocked_path:
            started.set()
            release.wait(30)
        return original_run(arguments)

    write_tool.run = slow_run
    return started, release


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


@pytest.mark.asyncio
async def test_interrupt_parallel_batch_records_real_results(workdir):
    client = FakeLLMClient(
        [
            [
                ToolCallEvent(
                    id="w1",
                    name="write",
                    arguments={"path": "a.txt", "content": "AAA"},
                ),
                ToolCallEvent(
                    id="w2",
                    name="write",
                    arguments={"path": "b.txt", "content": "BBB"},
                ),
            ]
        ]
    )
    agent = make_agent(workdir, client)
    started, release = _gate_write_tool(agent, "a.txt")

    events: list = []
    task = await _run_until(events, agent, started.is_set)
    agent.interrupt()
    # File tools are NOT cancelled: the blocked write runs to completion.
    release.set()
    await asyncio.wait_for(task, 15)

    assert (workdir / "a.txt").read_text() == "AAA"
    assert (workdir / "b.txt").read_text() == "BBB"
    results = {e.id: e.result for e in events if isinstance(e, ToolResultEvent)}
    assert results["w1"].success is True
    assert results["w2"].success is True
    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert interrupts[0].phase == "tool_batch"
    # Turn ended without another LLM iteration.
    assert len(client.calls) == 1
    # History pairing is intact: 2 tool calls, 2 real results.
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert {m.tool_call_id for m in tool_msgs} == {"w1", "w2"}


@pytest.mark.asyncio
async def test_interrupt_sequential_batch_placeholders_unstarted(workdir):
    config = Config()
    config.tools.parallel = False
    client = FakeLLMClient(
        [
            [
                ToolCallEvent(
                    id="w1",
                    name="write",
                    arguments={"path": "a.txt", "content": "AAA"},
                ),
                ToolCallEvent(
                    id="w2",
                    name="write",
                    arguments={"path": "b.txt", "content": "BBB"},
                ),
            ]
        ]
    )
    agent = make_agent(workdir, client, config)
    started, release = _gate_write_tool(agent, "a.txt")

    events: list = []
    task = await _run_until(events, agent, started.is_set)
    agent.interrupt()
    release.set()
    await asyncio.wait_for(task, 15)

    # The started call recorded its real result; the never-started one got
    # the placeholder and genuinely never touched disk.
    assert (workdir / "a.txt").read_text() == "AAA"
    assert not (workdir / "b.txt").exists()
    results = {e.id: e.result for e in events if isinstance(e, ToolResultEvent)}
    assert results["w1"].success is True
    assert results["w2"].success is False
    assert "未执行" in (results["w2"].error or "")
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert {m.tool_call_id for m in tool_msgs} == {"w1", "w2"}


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


@pytest.mark.asyncio
async def test_idle_interrupt_is_noop_and_does_not_leak(workdir):
    client = FakeLLMClient([[TextChunk(text="ok")]])
    agent = make_agent(workdir, client)

    agent.interrupt()  # idle: no turn, no compact — must not stick
    events = [e async for e in agent.run("hi")]

    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "ok"
    assert not any(isinstance(e, InterruptEvent) for e in events)
    records = read_trace(agent.trace.path)
    assert not any(r["type"] == "turn_interrupted" for r in records)
    turn_ends = [r for r in records if r["type"] == "turn_end"]
    assert turn_ends[-1]["status"] == "completed"
    assert agent.was_interrupted is False


@pytest.mark.asyncio
async def test_double_interrupt_logs_once(workdir):
    chunks = [TextChunk(text=f"c{i}") for i in range(10)]
    client = FakeLLMClient([chunks])
    agent = make_agent(workdir, client)

    events = []
    async for event in agent.run("hi"):
        events.append(event)
        if sum(isinstance(e, TextDelta) for e in events) == 2:
            agent.interrupt()
            agent.interrupt()

    records = read_trace(agent.trace.path)
    interrupted = [r for r in records if r["type"] == "turn_interrupted"]
    assert len(interrupted) == 1
    assert len([e for e in events if isinstance(e, InterruptEvent)]) == 1


# -- 6. Goal driver stops BEFORE the leftover-steer drain ------------------------


@pytest.mark.asyncio
async def test_driver_stops_on_interrupt_before_leftover_drain(
    workdir, monkeypatch
):
    verify_calls = []

    async def fake_verify(command, workdir, **kwargs):
        verify_calls.append(command)
        raise AssertionError("verify must not run after an interrupt")

    monkeypatch.setattr(goal_driver_module, "run_verify", fake_verify)

    chunks = [TextChunk(text=f"c{i}") for i in range(10)]
    client = FakeLLMClient([chunks])
    agent = make_agent(workdir, client)
    driver = GoalDriver(agent, workdir)
    driver.set_goal("g", verify_command="pytest")

    events = []
    interrupted = False
    async for event in driver.run("do it"):
        events.append(event)
        if not interrupted and sum(isinstance(e, TextDelta) for e in events) == 3:
            agent.steer("排队消息")
            agent.interrupt()
            interrupted = True

    assert verify_calls == []
    assert not any(isinstance(e, GoalVerifyStarted) for e in events)
    # The queued steer was NOT drained into a continuation round.
    assert not any(isinstance(e, SteerEvent) for e in events)
    assert agent.queued_count == 1
    assert driver.running is False
    # Goal stays active — the user called stop, not clear.
    assert driver.status().status == "active"
    assert len(client.calls) == 1
    assert "goal_interrupted" in trace_types(agent)


# -- 7. Leftover steers are injected by the NEXT run's head fallback -------------


@pytest.mark.asyncio
async def test_queued_steer_delivered_on_next_run_after_interrupt(workdir):
    client = FakeLLMClient(
        [
            [TextChunk(text=f"c{i}") for i in range(10)],
            [TextChunk(text="done")],
        ]
    )
    agent = make_agent(workdir, client)

    events = []
    interrupted = False
    async for event in agent.run("first"):
        events.append(event)
        if not interrupted and sum(isinstance(e, TextDelta) for e in events) == 3:
            agent.steer("插队")
            agent.interrupt()
            interrupted = True
    assert agent.queued_count == 1

    events2 = [e async for e in agent.run("second")]
    steers = [e for e in events2 if isinstance(e, SteerEvent)]
    assert len(steers) == 1
    assert steers[0].text == "插队"
    users = [m.content for m in agent.messages if m.role == "user"]
    assert users == ["first", "插队", "second"]
    records = read_trace(agent.trace.path)
    turn_ends = [r for r in records if r["type"] == "turn_end"]
    assert [t["status"] for t in turn_ends] == ["interrupted", "completed"]


# -- 8. Stalled stream (no chunk ever arrives) ------------------------------------


class StalledClient:
    def __init__(self):
        self.started = asyncio.Event()

    async def chat(self, messages, tools, on_request=None):
        self.started.set()
        await asyncio.sleep(3600)
        yield TextChunk(text="never")  # pragma: no cover - unreachable


@pytest.mark.asyncio
async def test_interrupt_stalled_stream(workdir):
    client = StalledClient()
    agent = make_agent(workdir, client)

    events: list = []

    async def consume():
        async for event in agent.run("hi"):
            events.append(event)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(client.started.wait(), 5)
    agent.interrupt()
    await asyncio.wait_for(task, 5)

    interrupts = [e for e in events if isinstance(e, InterruptEvent)]
    assert len(interrupts) == 1
    assert interrupts[0].phase == "llm_stream"
    # Nothing streamed: no assistant message is appended at all.
    assert agent.messages[-1].role == "user"
    records = read_trace(agent.trace.path)
    turn_ends = [r for r in records if r["type"] == "turn_end"]
    assert turn_ends[-1]["status"] == "interrupted"


# -- 9. Compact interruption and flag reset ---------------------------------------


class SetupThenSummaryClient:
    """First call answers normally; summary calls (tools=[]) follow a script."""

    def __init__(self):
        self.calls = []
        self.summary_started = asyncio.Event()
        self._summary_count = 0

    async def chat(self, messages, tools, on_request=None):
        self.calls.append((list(messages), tools))
        if tools:
            yield TextChunk(text="setup done")
            return
        self._summary_count += 1
        if self._summary_count == 1:
            self.summary_started.set()
            await asyncio.sleep(3600)
            yield TextChunk(text="never")  # pragma: no cover - unreachable
        else:
            yield TextChunk(text="摘要")


@pytest.mark.asyncio
async def test_interrupt_during_compact(workdir, monkeypatch):
    client = SetupThenSummaryClient()
    agent = make_agent(workdir, client)
    [e async for e in agent.run("setup")]
    before = list(agent.messages)

    # Force a valid split point regardless of history size.
    monkeypatch.setattr(
        "limbo.agent.find_split_point", lambda messages, keep_recent: 2
    )

    events: list = []

    async def consume():
        async for event in agent.compact(trigger="manual"):
            events.append(event)

    task = asyncio.create_task(consume())
    await asyncio.wait_for(client.summary_started.wait(), 5)
    agent.interrupt()
    await asyncio.wait_for(task, 5)

    compactions = [e for e in events if isinstance(e, CompactionEvent)]
    assert len(compactions) == 1
    assert compactions[0].compacted is False
    assert "打断" in (compactions[0].reason or "")
    assert any(
        isinstance(e, InterruptEvent) and e.phase == "compact" for e in events
    )
    # History untouched by the interrupted compaction.
    assert agent.messages == before

    # The interrupt flag must not leak into a later manual compact.
    events2 = [e async for e in agent.compact(trigger="manual")]
    assert any(
        isinstance(e, CompactionEvent) and e.compacted for e in events2
    )
