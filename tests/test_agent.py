"""Tests for the Agent conversation loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.agent import Agent, ErrorEvent
from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def chat(self, messages, tools):
        self.calls.append((messages, tools))
        for event in self.responses.pop(0):
            yield event


async def _collect(run):
    return [event async for event in run]


@pytest.mark.asyncio
async def test_agent_text_only_response(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))
    assert "".join([e.text for e in events if hasattr(e, "text")]) == "hello"


@pytest.mark.asyncio
async def test_agent_calls_tool_and_loops(workdir):
    (workdir / "main.py").write_text("x = 1\n")
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("read main.py"))
    assert any(hasattr(e, "name") and e.name == "read" for e in events)
    assert any(hasattr(e, "text") and e.text == "done" for e in events)


@pytest.mark.asyncio
async def test_agent_stops_on_confirmation(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("write x.txt"))
    result_events = [e for e in events if hasattr(e, "result")]
    assert len(result_events) == 1
    assert result_events[0].result.requires_confirmation is True
    assert agent._pending_tool is not None
    assert not (workdir / "x.txt").exists()


@pytest.mark.asyncio
async def test_agent_apply_tool_confirms(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("write x.txt"))

    result = await agent.apply_tool("write", {"path": "x.txt", "content": "hi"})
    assert result.success is True
    assert (workdir / "x.txt").read_text() == "hi"
    assert agent._pending_tool is None


@pytest.mark.asyncio
async def test_agent_apply_tool_rejects_mismatch(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("write x.txt"))

    result = await agent.apply_tool("write", {"path": "x.txt", "content": "different"})
    assert result.success is False
    assert "does not match" in result.error
    assert agent._pending_tool is not None
    assert not (workdir / "x.txt").exists()


@pytest.mark.asyncio
async def test_agent_apply_tool_requires_pending_tool(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("hi"))

    result = await agent.apply_tool("write", {"path": "x.txt", "content": "hi"})
    assert result.success is False
    assert "No tool is pending" in result.error


@pytest.mark.asyncio
async def test_agent_continue_after_confirmation_resumes_loop(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("write x.txt"))
    result_events = [e for e in events if hasattr(e, "result")]
    assert len(result_events) == 1
    assert result_events[0].result.requires_confirmation is True

    apply_result = await agent.apply_tool("write", {"path": "x.txt", "content": "hi"})
    assert apply_result.success is True
    assert (workdir / "x.txt").read_text() == "hi"

    continuation = await _collect(agent.continue_after_confirmation())
    assert any(hasattr(e, "text") and e.text == "done" for e in continuation)
    assert agent._pending_tool is None
    assert agent._confirmation_applied is False


@pytest.mark.asyncio
async def test_agent_saves_session(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    session_dir = workdir / "sessions"
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=session_dir,
    )

    await _collect(agent.run("hi"))

    assert session_dir.exists()
    files = list(session_dir.iterdir())
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) >= 2
    assert '"role":"system"' in lines[0]
    assert any('"role":"user"' in line for line in lines)


@pytest.mark.asyncio
async def test_agent_error_event_on_llm_failure(workdir):
    class FailingLLMClient:
        async def chat(self, messages, tools):
            raise RuntimeError("network down")
            if False:
                yield TextChunk(text="")

    cfg = Config()
    agent = Agent(
        config=cfg,
        llm_client=FailingLLMClient(),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert "network down" in error_events[0].message


@pytest.mark.asyncio
async def test_agent_iteration_count_resets_per_turn(workdir):
    cfg = Config()
    cfg.llm.max_iterations = 1
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
        [TextChunk(text="second turn")],
    ])
    (workdir / "main.py").write_text("x = 1\n")
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("turn one"))
    assert agent._iteration_count == 1
    assert any(hasattr(e, "name") and e.name == "read" for e in events)

    events = await _collect(agent.run("turn two"))
    assert agent._iteration_count == 1
    assert any(hasattr(e, "text") and e.text == "second turn" for e in events)


@pytest.mark.asyncio
async def test_agent_reject_pending_tool_clears_state(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("write x.txt"))
    assert agent._pending_tool is not None

    agent.reject_pending_tool()
    assert agent._pending_tool is None
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert any(
        m.tool_call_id == "c1" and "rejected" in (m.content or "").lower()
        for m in tool_messages
    )


@pytest.mark.asyncio
async def test_agent_new_turn_clears_stale_pending_tool(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"})],
        [TextChunk(text="fresh turn")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("write x.txt"))
    assert agent._pending_tool is not None

    events = await _collect(agent.run("new turn"))
    assert agent._pending_tool is None
    assert any(hasattr(e, "text") and e.text == "fresh turn" for e in events)
    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert any(
        m.tool_call_id == "c1" and "superseded" in (m.content or "").lower()
        for m in tool_messages
    )


@pytest.mark.asyncio
async def test_agent_multi_tool_confirmation_appends_placeholders(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"}),
            ToolCallEvent(id="c2", name="write", arguments={"path": "x.txt", "content": "hi"}),
            ToolCallEvent(id="c3", name="ls", arguments={"path": "."}),
        ],
    ])
    (workdir / "main.py").write_text("x = 1\n")
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("multi"))
    # The second tool (write) requires confirmation, so the loop stops there.
    result_events = [e for e in events if hasattr(e, "result")]
    assert len(result_events) == 2
    assert result_events[1].result.requires_confirmation is True
    assert agent._pending_tool is not None
    assert agent._pending_tool["id"] == "c2"

    # Every assistant tool_call must have a matching tool result message.
    assistant = [m for m in agent.messages if m.role == "assistant" and m.tool_calls]
    assert len(assistant) == 1
    tool_call_ids = {tc["id"] for tc in assistant[0].tool_calls or []}
    tool_result_ids = {m.tool_call_id for m in agent.messages if m.role == "tool"}
    assert tool_call_ids == tool_result_ids

    # The pending tool and remaining calls have placeholder results.
    pending_placeholder = next(
        m for m in agent.messages if m.role == "tool" and m.tool_call_id == "c2"
    )
    assert "pending" in (pending_placeholder.content or "").lower()
    remaining_placeholder = next(
        m for m in agent.messages if m.role == "tool" and m.tool_call_id == "c3"
    )
    assert "pending" in (remaining_placeholder.content or "").lower()

    # After confirmation the pending placeholder is replaced with the real result.
    apply_result = await agent.apply_tool(
        "write", {"path": "x.txt", "content": "hi"}
    )
    assert apply_result.success is True
    c2_message = next(
        m for m in agent.messages if m.role == "tool" and m.tool_call_id == "c2"
    )
    assert "wrote" in (c2_message.content or "").lower()
