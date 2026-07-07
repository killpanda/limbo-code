"""Tests for the Agent conversation loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append((messages, tools))
        for event in self.responses.pop(0):
            yield event


def test_agent_text_only_response(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=workdir)

    events = list(agent.run("hi"))
    assert "".join([e.text for e in events if hasattr(e, "text")]) == "hello"


def test_agent_calls_tool_and_loops(workdir):
    (workdir / "main.py").write_text("x = 1\n")
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
        [TextChunk(text="done")],
    ])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=workdir)

    events = list(agent.run("read main.py"))
    assert any(hasattr(e, "name") and e.name == "read" for e in events)
    assert any(hasattr(e, "text") and e.text == "done" for e in events)


def test_agent_stops_on_confirmation(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
    ])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=workdir)

    events = list(agent.run("write x.txt"))
    result_events = [e for e in events if hasattr(e, "result")]
    assert len(result_events) == 1
    assert result_events[0].result.requires_confirmation is True
    assert agent._pending_tool is not None
    assert not (workdir / "x.txt").exists()


def test_agent_apply_tool_confirms(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
    ])
    agent = Agent(config=cfg, llm_client=fake_llm, workdir=workdir)
    list(agent.run("write x.txt"))

    result = agent.apply_tool("write", {"path": "x.txt", "content": "hi"})
    assert result.success is True
    assert (workdir / "x.txt").read_text() == "hi"
    assert agent._pending_tool is None
