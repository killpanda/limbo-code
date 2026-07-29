"""Tests for the Agent conversation loop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from limbo.agent import (
    Agent,
    ErrorEvent,
    TextDelta,
    ToolCallRequest,
    ToolResultEvent,
)
from limbo.config import Config
from limbo.llm.retry import LLMHttpError
from limbo.models import CompletionMeta, Message, TextChunk, ToolCallEvent, ToolResult
from limbo.trace import read_trace


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append((messages, tools))
        if on_request is not None:
            on_request(
                {
                    "model": "fake-model",
                    "messages": [m.model_dump() for m in messages],
                    "tools": tools,
                }
            )
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
    assert "".join([e.text for e in events if isinstance(e, TextDelta)]) == "hello"


def test_system_prompt_includes_skills_catalog(workdir):
    skill_dir = workdir / ".agents" / "skills" / "tdd"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: tdd\ndescription: Test-driven development workflow.\n---\n\nDo TDD.\n"
    )
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([]),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    prompt = agent.messages[0].content
    assert "<available_skills>" in prompt
    assert "<name>tdd</name>" in prompt
    assert "<description>Test-driven development workflow.</description>" in prompt
    assert "<location>" in prompt
    assert "Do TDD." not in prompt  # body stays on disk


def test_system_prompt_skips_catalog_when_no_skills(workdir, monkeypatch):
    # Point the user skills dir at an empty location so only the (empty)
    # project dir is scanned.
    monkeypatch.setattr("limbo.agent.discover_skills", lambda workdir: [])
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([]),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    assert "<available_skills>" not in agent.messages[0].content


def test_system_prompt_wraps_agents_md_in_xml_boundary(workdir):
    (workdir / "AGENTS.md").write_text("# Project rules\n")
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([]),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    prompt = agent.messages[0].content
    assert "<project_context>" in prompt
    assert f'<project_instructions path="{workdir / "AGENTS.md"}">' in prompt
    assert "# Project rules" in prompt
    assert "</project_context>" in prompt


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
    assert any(isinstance(e, ToolCallRequest) and e.name == "read" for e in events)
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)


@pytest.mark.asyncio
async def test_agent_executes_write_tool_immediately(workdir):
    """Tools run without confirmation: write applies during the turn."""
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
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.success is True
    assert (workdir / "x.txt").read_text() == "hi"
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)

    tool_message = next(
        m for m in agent.messages if m.role == "tool" and m.tool_call_id == "c1"
    )
    assert "wrote" in (tool_message.content or "").lower()


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
    files = list(session_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    assert len(lines) >= 2
    # First line is the session meta record, followed by messages.
    assert json.loads(lines[0])["type"] == "meta"
    assert '"role":"system"' in lines[1]
    assert any('"role":"user"' in line for line in lines)


@pytest.mark.asyncio
async def test_agent_save_session_failure_is_ignored(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    agent._save_session_sync = lambda: (_ for _ in ()).throw(RuntimeError("disk full"))

    with pytest.warns(UserWarning, match="Failed to save session"):
        events = await _collect(agent.run("hi"))

    assert any(isinstance(e, TextDelta) and e.text == "hello" for e in events)


@pytest.mark.asyncio
async def test_agent_error_event_on_llm_failure(workdir):
    class FailingLLMClient:
        async def chat(self, messages, tools, on_request=None):
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
    assert any(isinstance(e, ToolCallRequest) and e.name == "read" for e in events)

    events = await _collect(agent.run("turn two"))
    assert agent._iteration_count == 1
    assert any(isinstance(e, TextDelta) and e.text == "second turn" for e in events)


@pytest.mark.asyncio
async def test_agent_multi_tool_turn_executes_all_calls(workdir):
    """Every tool call in a multi-tool turn executes, without pausing."""
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="read", arguments={"path": "a.txt"}),
            ToolCallEvent(id="c2", name="write", arguments={"path": "x.txt", "content": "hi"}),
            ToolCallEvent(id="c3", name="read", arguments={"path": "b.txt"}),
        ],
        [TextChunk(text="done")],
    ])
    (workdir / "a.txt").write_text("aaa\n")
    (workdir / "b.txt").write_text("bbb\n")
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("multi"))
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert {e.id for e in result_events} == {"c1", "c2", "c3"}
    assert all(e.result.success for e in result_events)
    assert (workdir / "x.txt").read_text() == "hi"
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)

    # Every assistant tool_call must have a matching tool result message.
    assistant = [m for m in agent.messages if m.role == "assistant" and m.tool_calls]
    assert len(assistant) == 1
    tool_call_ids = {tc["id"] for tc in assistant[0].tool_calls or []}
    tool_result_ids = {m.tool_call_id for m in agent.messages if m.role == "tool"}
    assert tool_call_ids == tool_result_ids

    # Results land in history in assistant source order, not completion order.
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_agent_tool_crash_becomes_error_result_and_loop_continues(workdir):
    """A crashed tool yields an error ToolResult; siblings and the loop go on."""
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

    async def crashing_execute(name, arguments):
        raise RuntimeError("tool exploded")

    agent.registry.execute = crashing_execute

    events = await _collect(agent.run("read main.py"))
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.success is False
    assert "tool exploded" in (result_events[0].result.error or "")
    # The loop continued and produced the final text response.
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)
    assert not any(isinstance(e, ErrorEvent) for e in events)


@pytest.mark.asyncio
async def test_agent_tool_crash_does_not_cancel_siblings(workdir):
    """A crashed call must not cancel its siblings; all calls get results."""
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"}),
            ToolCallEvent(id="c2", name="write", arguments={"path": "x.txt", "content": "hi"}),
        ],
        [TextChunk(text="done")],
    ])
    (workdir / "main.py").write_text("x = 1\n")
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    original_execute = agent.registry.execute

    async def execute_with_crash(name, arguments):
        if name == "read":
            raise RuntimeError("tool exploded")
        return await original_execute(name, arguments)

    agent.registry.execute = execute_with_crash

    events = await _collect(agent.run("multi"))
    result_events = {e.id: e for e in events if isinstance(e, ToolResultEvent)}
    assert result_events["c1"].result.success is False
    assert "tool exploded" in (result_events["c1"].result.error or "")
    assert result_events["c2"].result.success is True

    assistant = [m for m in agent.messages if m.role == "assistant" and m.tool_calls]
    assert len(assistant) == 1
    tool_call_ids = {tc["id"] for tc in assistant[0].tool_calls or []}
    tool_result_ids = {m.tool_call_id for m in agent.messages if m.role == "tool"}
    assert tool_call_ids == tool_result_ids

    crashed_message = next(
        m for m in agent.messages if m.role == "tool" and m.tool_call_id == "c1"
    )
    assert "tool exploded" in (crashed_message.content or "")
    sibling_message = next(
        m for m in agent.messages if m.role == "tool" and m.tool_call_id == "c2"
    )
    assert "hi" not in (sibling_message.content or "").lower()  # it's the write ack
    assert "wrote" in (sibling_message.content or "").lower()
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)


@pytest.mark.asyncio
async def test_agent_max_iterations_cancels_tool_calls(workdir):
    cfg = Config()
    cfg.llm.max_iterations = 1
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
    ])
    (workdir / "main.py").write_text("x = 1\n")
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    executed = []
    original_execute = agent.registry.execute

    async def tracking_execute(name, arguments):
        executed.append((name, arguments))
        return await original_execute(name, arguments)

    agent.registry.execute = tracking_execute

    events = await _collect(agent.run("read main.py"))

    # The tool must not be executed once the iteration limit is reached.
    assert len(executed) == 0

    assistant = [m for m in agent.messages if m.role == "assistant" and m.tool_calls]
    assert len(assistant) == 1
    assert assistant[0].tool_calls[0]["id"] == "c1"

    tool_messages = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c1"
    assert "maximum" in (tool_messages[0].content or "").lower()

    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert "maximum" in error_events[0].message.lower()


@pytest.mark.asyncio
async def test_agent_rewrites_session_on_subsequent_turns(workdir):
    cfg = Config()
    fake_llm = FakeLLMClient([
        [TextChunk(text="first")],
        [TextChunk(text="second")],
    ])
    session_dir = workdir / "sessions"
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=session_dir,
    )

    await _collect(agent.run("turn one"))
    await _collect(agent.run("turn two"))

    files = list(session_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().strip().splitlines()
    roles = [
        json.loads(line).get("role", "meta") for line in lines
    ]
    assert roles.count("system") == 1
    assert roles.count("user") == 2
    assert "assistant" in roles


@pytest.mark.asyncio
async def test_agent_save_session_is_atomic(workdir, monkeypatch):
    """Session writes must use a temp file and atomic replace."""
    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hello")]])
    session_dir = workdir / "sessions"
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=session_dir,
    )

    replaced: list[tuple[Path, Path]] = []
    original_replace = os.replace

    def tracking_replace(src: str | Path, dst: str | Path) -> None:
        replaced.append((Path(src), Path(dst)))
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)

    await _collect(agent.run("hi"))

    assert len(replaced) == 1
    src, dst = replaced[0]
    assert src.suffix == ".tmp"
    assert dst == agent._session_file
    # The temp file should not be left behind after a successful save.
    assert not src.exists()
    assert dst.exists()


# --- session resume -------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_saves_meta_with_title(workdir):
    from limbo.sessions import list_sessions, load_session

    cfg = Config()
    fake_llm = FakeLLMClient([[TextChunk(text="hi")]])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    await _collect(agent.run("fix the   grep bug please"))

    sessions = list_sessions(workdir / "sessions")
    assert len(sessions) == 1
    assert sessions[0].title == "fix the grep bug please"
    assert sessions[0].workdir == str(workdir.resolve())
    assert sessions[0].id == agent.session_id

    _, messages = load_session(sessions[0].path)
    assert messages[0].role == "system"
    assert any(m.role == "user" for m in messages)


@pytest.mark.asyncio
async def test_agent_resume_continues_same_file_and_history(workdir):
    from limbo.sessions import latest_session, list_sessions

    session_dir = workdir / "sessions"
    cfg = Config()
    agent1 = Agent(
        config=cfg,
        llm_client=FakeLLMClient([[TextChunk(text="answer one")]]),
        workdir=workdir,
        session_dir=session_dir,
    )
    await _collect(agent1.run("first question"))
    session_path = latest_session(session_dir)

    agent2 = Agent(
        config=cfg,
        llm_client=FakeLLMClient([[TextChunk(text="answer two")]]),
        workdir=workdir,
        session_dir=session_dir,
        resume=session_path,
    )
    assert agent2.session_id == agent1.session_id
    await _collect(agent2.run("second question"))

    # No new session file was created; history was sent to the LLM.
    assert len(list_sessions(session_dir)) == 1
    sent_messages = agent2.llm_client.calls[0][0]
    assert sent_messages[0].role == "system"
    contents = [m.content for m in sent_messages]
    assert "first question" in contents
    assert "answer one" in contents
    assert "second question" in contents


@pytest.mark.asyncio
async def test_agent_resume_regenerates_system_message(workdir):
    from limbo.sessions import SessionMeta, save_session

    session_dir = workdir / "sessions"
    session_dir.mkdir()
    path = session_dir / "abc123.jsonl"
    save_session(
        path,
        SessionMeta(id="abc123"),
        [
            Message(role="system", content="OLD SYSTEM PROMPT"),
            Message(role="user", content="hi"),
        ],
    )

    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([[TextChunk(text="ok")]]),
        workdir=workdir,
        session_dir=session_dir,
        resume=path,
    )
    await _collect(agent.run("again"))

    sent_messages = agent.llm_client.calls[0][0]
    system_messages = [m for m in sent_messages if m.role == "system"]
    assert len(system_messages) == 1
    assert "OLD SYSTEM PROMPT" not in (system_messages[0].content or "")


@pytest.mark.asyncio
async def test_agent_resume_repairs_dangling_tool_calls(workdir):
    from limbo.models import Message
    from limbo.sessions import SessionMeta, save_session

    session_dir = workdir / "sessions"
    session_dir.mkdir()
    path = session_dir / "dangling.jsonl"
    save_session(
        path,
        SessionMeta(id="dangling"),
        [
            Message(role="user", content="do something"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            ),
            # Crash before the tool result was recorded.
        ],
    )

    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([[TextChunk(text="recovered")]]),
        workdir=workdir,
        session_dir=session_dir,
        resume=path,
    )
    await _collect(agent.run("continue"))

    sent_messages = agent.llm_client.calls[0][0]
    tool_messages = [m for m in sent_messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "c1"
    # The placeholder must immediately follow its assistant message.
    assistant_idx = next(
        i for i, m in enumerate(sent_messages) if m.tool_calls
    )
    assert sent_messages[assistant_idx + 1].role == "tool"


@pytest.mark.asyncio
async def test_agent_stores_reasoning_on_assistant_message(workdir):
    from limbo.agent import ThinkingDelta
    from limbo.models import ThinkingChunk

    cfg = Config()
    fake_llm = FakeLLMClient(
        [[ThinkingChunk(text="hmm "), ThinkingChunk(text="thinking"), TextChunk(text="answer")]]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))
    thinking = "".join(e.text for e in events if isinstance(e, ThinkingDelta))
    assert thinking == "hmm thinking"

    last = agent.messages[-1]
    assert last.role == "assistant"
    assert last.reasoning == "hmm thinking"
    assert last.content == "answer"


# --- trace logging -------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_trace_records_full_turn(workdir):
    (workdir / "main.py").write_text("x = 1\n")
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_cache_hit_tokens": 80,
        "prompt_cache_miss_tokens": 20,
    }
    fake_llm = FakeLLMClient([
        [
            CompletionMeta(
                usage=usage, finish_reason="tool_calls", ttft=0.01, duration=0.5
            ),
            ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"}),
        ],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=Config(),
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    await _collect(agent.run("read main.py"))

    records = read_trace(agent.trace.path)
    types = [r["type"] for r in records]
    assert types == [
        "session_start",
        "user_message",
        "llm_request",
        "llm_response",
        "tool_call",
        "tool_result",
        "llm_request",
        "llm_response",
        "turn_end",
    ]

    start = records[0]
    assert start["config"]["model"] == "deepseek-chat"
    assert start["resumed"] is False
    # The API key must never appear in the trace.
    assert "api_key" not in json.dumps(records)

    assert records[1]["content"] == "read main.py"

    request = records[2]
    assert request["turn"] == 1 and request["iteration"] == 1
    assert request["body"]["model"] == "fake-model"
    assert request["body"]["messages"][0]["role"] == "system"
    assert request["body"]["tools"]  # full tool definitions included

    response = records[3]
    assert response["usage"] == usage
    assert response["cached_tokens"] == 80
    assert response["finish_reason"] == "tool_calls"
    assert response["tool_calls"] == [{"id": "c1", "name": "read"}]

    tool_call, tool_result = records[4], records[5]
    assert tool_call["name"] == "read"
    assert tool_call["arguments"] == {"path": "main.py"}
    assert tool_result["success"] is True
    assert tool_result["duration"] >= 0
    assert "x = 1" in tool_result["output"]

    turn_end = records[-1]
    assert turn_end["status"] == "completed"
    assert turn_end["iterations"] == 2


@pytest.mark.asyncio
async def test_agent_trace_records_llm_error(workdir):
    class FailingLLMClient:
        async def chat(self, messages, tools, on_request=None):
            raise RuntimeError("network down")
            if False:
                yield TextChunk(text="")

    agent = Agent(
        config=Config(),
        llm_client=FailingLLMClient(),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("hi"))

    records = read_trace(agent.trace.path)
    error = next(r for r in records if r["type"] == "llm_error")
    assert error["exception_type"] == "RuntimeError"
    assert "network down" in error["error"]
    assert "network down" in error["traceback"]


@pytest.mark.asyncio
async def test_agent_friendly_error_on_rate_limit(workdir):
    """429-class failures surface the friendly hint; trace keeps the raw error."""

    class RateLimitedLLMClient:
        async def chat(self, messages, tools, on_request=None):
            raise LLMHttpError(429, "Error code: 429 - rate limit exceeded")
            if False:
                yield TextChunk(text="")

    agent = Agent(
        config=Config(),
        llm_client=RateLimitedLLMClient(),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert "稍后重发" in error_events[0].message

    records = read_trace(agent.trace.path)
    error = next(r for r in records if r["type"] == "llm_error")
    assert error["exception_type"] == "LLMHttpError"
    assert "429" in error["error"]


@pytest.mark.asyncio
async def test_agent_friendly_message_fallback_for_unknown_error(workdir):
    """friendly_message returns None for unclassified errors -> raw text kept."""

    class FailingLLMClient:
        async def chat(self, messages, tools, on_request=None):
            raise ValueError("weird failure")
            if False:
                yield TextChunk(text="")

    agent = Agent(
        config=Config(),
        llm_client=FailingLLMClient(),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))
    error_events = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].message == "LLM error: weird failure"


@pytest.mark.asyncio
async def test_agent_trace_records_tool_crash(workdir):
    fake_llm = FakeLLMClient([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
    ])
    agent = Agent(
        config=Config(),
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    async def crashing_execute(name, arguments):
        raise RuntimeError("tool exploded")

    agent.registry.execute = crashing_execute
    await _collect(agent.run("read main.py"))

    records = read_trace(agent.trace.path)
    result = next(r for r in records if r["type"] == "tool_result")
    assert result["success"] is False
    assert result["exception_type"] == "RuntimeError"
    assert "tool exploded" in result["traceback"]


@pytest.mark.asyncio
async def test_agent_trace_file_does_not_pollute_session_listing(workdir):
    from limbo.sessions import find_session, list_sessions

    session_dir = workdir / "sessions"
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([[TextChunk(text="hi")]]),
        workdir=workdir,
        session_dir=session_dir,
    )
    await _collect(agent.run("hello"))

    assert agent.trace.path.parent.name == "traces"
    assert agent.trace.path.exists()
    sessions = list_sessions(session_dir)
    assert len(sessions) == 1
    # find_session must not see the trace file as an ambiguous match.
    assert find_session(session_dir, agent.session_id) == agent._session_file


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_parallel_tool_calls_run_concurrently(workdir):
    """Two slow tool calls must overlap: wall time < serial execution."""
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="read", arguments={"path": "a.txt"}),
            ToolCallEvent(id="c2", name="read", arguments={"path": "b.txt"}),
        ],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    async def slow_execute(name, arguments):
        await asyncio.sleep(0.3)
        return ToolResult(success=True, output=f"ok:{name}")

    agent.registry.execute = slow_execute

    start = time.monotonic()
    events = await _collect(agent.run("read both"))
    elapsed = time.monotonic() - start

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(e.result.success for e in result_events)
    # Serial execution would take ~0.6s; concurrent should be well under.
    assert elapsed < 0.55


@pytest.mark.asyncio
async def test_agent_parallel_events_completion_order_history_source_order(workdir):
    """Events stream in completion order; history records in source order."""
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="read", arguments={"path": "slow.txt"}),
            ToolCallEvent(id="c2", name="read", arguments={"path": "fast.txt"}),
        ],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    async def variably_slow_execute(name, arguments):
        if arguments["path"] == "slow.txt":
            await asyncio.sleep(0.3)
        return ToolResult(success=True, output=f"ok:{arguments['path']}")

    agent.registry.execute = variably_slow_execute

    events = await _collect(agent.run("read both"))
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    # c2 finishes first even though c1 was requested first.
    assert [e.id for e in result_events] == ["c2", "c1"]

    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_agent_parallel_disabled_falls_back_to_sequential(workdir):
    cfg = Config()
    cfg.tools.parallel = False
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="read", arguments={"path": "slow.txt"}),
            ToolCallEvent(id="c2", name="read", arguments={"path": "fast.txt"}),
        ],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    async def variably_slow_execute(name, arguments):
        if arguments["path"] == "slow.txt":
            await asyncio.sleep(0.2)
        return ToolResult(success=True, output="ok")

    agent.registry.execute = variably_slow_execute

    events = await _collect(agent.run("read both"))
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [e.id for e in result_events] == ["c1", "c2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
async def test_agent_length_stop_fails_whole_batch_without_executing(
    workdir, finish_reason
):
    """Truncated responses fail all tool calls without executing them.

    OpenAI signals output truncation as ``length``; Anthropic as
    ``max_tokens`` (its stop_reason is passed through verbatim).
    """
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="write", arguments={"path": "x.txt", "content": "hi"}),
            ToolCallEvent(id="c2", name="read", arguments={"path": "y.txt"}),
            CompletionMeta(finish_reason=finish_reason),
        ],
        [TextChunk(text="done")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    executed = []
    original_execute = agent.registry.execute

    async def spy_execute(name, arguments):
        executed.append(name)
        return await original_execute(name, arguments)

    agent.registry.execute = spy_execute

    events = await _collect(agent.run("go"))
    assert executed == []

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert {e.id for e in result_events} == {"c1", "c2"}
    assert all(not e.result.success for e in result_events)
    assert all("truncated" in (e.result.error or "") for e in result_events)

    # Every call still got a matching tool message, and the loop continued.
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["c1", "c2"]
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)

    records = read_trace(agent.trace.path)
    assert any(r.get("kind") == "length_stop" for r in records if r["type"] == "error")


@pytest.mark.asyncio
async def test_agent_same_file_write_and_edit_do_not_interleave(workdir):
    """Concurrent write+edit on the same file serialize via the mutation queue."""
    cfg = Config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(id="c1", name="write", arguments={"path": "f.txt", "content": "one"}),
            ToolCallEvent(
                id="c2",
                name="edit",
                arguments={"path": "f.txt", "old_text": "seed", "new_text": "two"},
            ),
        ],
        [TextChunk(text="done")],
    ])
    (workdir / "f.txt").write_text("seed")
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("mutate"))
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    # Both calls completed; the file ends in one of the two consistent states
    # (serialized), never an interleaved corruption.
    assert (workdir / "f.txt").read_text() in {"one", "two"}
