"""Tests for the headless CLI frontend (RFC LIM-60, design/rfc-headless-cli.md).

Seam under test: ``HeadlessFrontend.run(inputs, out)`` — a real Agent driven
by a scripted FakeLLMClient (same pattern as test_pump.py), a recording fake
reporter for lifecycle assertions, and an async iterator of InputEvents as
stdin. Output is collected into a list of written chunks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import limbo.pump as pump_module
from limbo.agent import Agent
from limbo.config import Config
from limbo.headless import (
    Eof,
    HeadlessFrontend,
    Interrupt,
    Submit,
)
from limbo.models import TextChunk
from limbo.pump import TurnPump


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append([m for m in messages])
        response = self.responses.pop(0)
        if callable(response):
            response = await response()
        for event in response:
            yield event


class RecordingReporter:
    """IntegrationReporter-shaped recorder (release included)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def report(self, state, *, message=None, session_id=None, session_path=None):
        self.calls.append(("report", state, message))

    def report_session(self, session_id, session_path=None):
        self.calls.append(("session", session_id))

    def release(self):
        self.calls.append(("release",))

    @property
    def states(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "report"]


def make_frontend(workdir: Path, responses, reporter=None, json_mode=False):
    """Build a frontend wired to a real Agent + scripted LLM."""
    config = Config()
    agent = Agent(
        config=config,
        llm_client=FakeLLMClient(responses),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    pump = TurnPump(agent, workdir)
    reporter = reporter or RecordingReporter()
    chunks: list[str] = []

    def make_agent(client, resume=None):
        return Agent(
            config=config,
            llm_client=client,
            workdir=workdir,
            session_dir=workdir / "sessions",
            resume=resume,
        )

    frontend = HeadlessFrontend(
        config=config,
        agent=agent,
        pump=pump,
        llm_client=agent.llm_client,
        reporters=reporter,
        out=chunks.append,
        agent_factory=make_agent,
        pump_factory=lambda a: TurnPump(a, workdir),
        echo_input=not json_mode,
        json_mode=json_mode,
    )
    return frontend, chunks, reporter


async def feed(events):
    for event in events:
        yield event


def output_of(chunks) -> str:
    return "".join(chunks)


async def wait_until(cond, timeout=2.0):
    async def _poll():
        while not cond():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


# -- single turn ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_echoes_streams_and_reports(workdir):
    frontend, chunks, reporter = make_frontend(
        workdir, [[TextChunk(text="hello world")]]
    )

    rc = await frontend.run(feed([Submit("你好"), Eof()]))

    assert rc == 0
    out = output_of(chunks)
    assert "❯ 你好" in out  # pipe-mode echo of the submission
    assert "hello world" in out  # streamed assistant text
    assert "── limbo idle ·" in out  # turn-end marker
    # Lifecycle: session -> idle -> working -> idle, released at the end.
    assert reporter.states == ["idle", "working", "done"]
    assert reporter.calls[0][0] == "session"
    assert reporter.calls[-1] == ("release",)


# -- tool rendering ------------------------------------------------------------


# -- steer while busy ----------------------------------------------------------


# -- interrupt -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_cancels_stream(workdir):
    started = asyncio.Event()

    async def hangs():
        started.set()
        await asyncio.Event().wait()  # never completes; interrupt cancels it
        return []

    frontend, chunks, reporter = make_frontend(workdir, [hangs])

    async def script():
        yield Submit("长跑任务")
        await started.wait()
        yield Interrupt(hard=False)  # Esc
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    assert "⏹ 已打断" in output_of(chunks)
    assert reporter.states == ["idle", "working", "idle"]


# -- commands refused while busy ------------------------------------------------


# -- EOF while busy -------------------------------------------------------------


# -- goal: proposal -> blocked -> confirm ---------------------------------------

def stub_verify(monkeypatch, results):
    queue = list(results)
    seen = []

    async def fake(command, workdir, **kwargs):
        seen.append(command)
        return queue.pop(0)

    monkeypatch.setattr(pump_module, "run_verify", fake)
    return seen


# -- /compact, /model, /new -----------------------------------------------------


# -- submission / verify-answer parsing (pure) -----------------------------------


# -- terminal line parser (pure bytes) -------------------------------------------


# -- layering lock: the headless frontend stays textual-free ---------------------


