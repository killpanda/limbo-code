"""Tests for the Agent conversation loop."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from limbo.agent import (
    ATTACHMENT_INLINE_MAX_BYTES,
    Agent,
    CompactionEvent,
    ErrorEvent,
    TextDelta,
    ToolCallRequest,
    ToolResultEvent,
    _extract_cached_tokens,
)
from limbo.compaction import SUMMARY_TAG
from limbo.config import CompactionSettings, Config
from limbo.llm.retry import LLMHttpError
from limbo.models import (
    Attachment,
    CompletionMeta,
    Message,
    TextChunk,
    ToolCallEvent,
    ToolResult,
)
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

    _, messages, _ = load_session(sessions[0].path)
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


# -- auto-compaction (LIM-14) ---------------------------------------------------


def _compact_cfg(**overrides) -> Config:
    # 1000 is the validator floor; tests pad messages so history exceeds it.
    defaults = {"keep_recent_tokens": 1000}
    defaults.update(overrides)
    return Config(compaction=CompactionSettings(**defaults))


def _seed_history(agent: Agent) -> None:
    """Give the agent enough history for find_split_point to work with."""
    agent.messages.append(Message(role="user", content="a" * 3000))
    agent.messages.append(Message(role="assistant", content="b" * 3000))


def _big_usage(tokens: int = 120_000) -> CompletionMeta:
    return CompletionMeta(usage={"prompt_tokens": tokens})


@pytest.mark.asyncio
async def test_auto_compaction_triggers_before_next_call(workdir):
    cfg = _compact_cfg()
    fake_llm = FakeLLMClient(
        [
            [
                TextChunk(text="working"),
                ToolCallEvent(id="c1", name="ls", arguments={}),
                _big_usage(),
            ],
            [TextChunk(text="SUMMARY"), CompletionMeta()],  # the summary call
            [TextChunk(text="final"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    _seed_history(agent)

    events = await _collect(agent.run("x" * 2000))

    compacted = [e for e in events if isinstance(e, CompactionEvent)]
    assert len(compacted) == 1
    assert compacted[0].compacted and compacted[0].trigger == "auto"
    # The summary call carried no tools...
    assert fake_llm.calls[1][1] == []
    # ...and the next real request went out with the compacted history.
    next_messages = fake_llm.calls[2][0]
    assert next_messages[1].content.startswith(f"<{SUMMARY_TAG}>")
    assert "SUMMARY" in next_messages[1].content
    # The turn completed normally after compacting.
    assert "final" in "".join(e.text for e in events if isinstance(e, TextDelta))


@pytest.mark.asyncio
async def test_auto_compaction_catches_stale_usage_between_turns(workdir):
    """P1-1: a huge paste between turns must trigger compaction even though
    the last reported prompt_tokens was small."""
    cfg = _compact_cfg()
    fake_llm = FakeLLMClient(
        [
            [TextChunk(text="ok"), CompletionMeta(usage={"prompt_tokens": 1000})],
            [TextChunk(text="SUMMARY"), CompletionMeta()],
            [TextChunk(text="answer"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("hi"))

    # Simulate a huge paste landing between turns (before the next run).
    agent.messages.append(Message(role="user", content="z" * 500_000))
    events = await _collect(agent.run("please continue"))

    compacted = [e for e in events if isinstance(e, CompactionEvent)]
    assert compacted and compacted[0].compacted
    # Compaction happened before this turn's first real LLM call.
    first_real_call = fake_llm.calls[2][0]
    assert first_real_call[1].content.startswith(f"<{SUMMARY_TAG}>")


@pytest.mark.asyncio
async def test_auto_compaction_summary_failure_keeps_history_and_guards_turn(
    workdir,
):
    class SummaryFailingClient:
        def __init__(self):
            self.calls = []
            self.summary_attempts = 0

        async def chat(self, messages, tools, on_request=None):
            self.calls.append((messages, tools))
            if not tools:
                self.summary_attempts += 1
                raise RuntimeError("summary boom")
            if len(self.calls) == 1:
                for event in [
                    TextChunk(text="working"),
                    ToolCallEvent(id="c1", name="ls", arguments={}),
                    _big_usage(),
                ]:
                    yield event
            else:
                for event in [
                    TextChunk(text="final"),
                    _big_usage(),  # still over threshold
                ]:
                    yield event

    cfg = _compact_cfg()
    client = SummaryFailingClient()
    agent = Agent(
        config=cfg,
        llm_client=client,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    _seed_history(agent)

    events = await _collect(agent.run("x" * 2000))

    compacted = [e for e in events if isinstance(e, CompactionEvent)]
    assert compacted and not compacted[0].compacted
    assert "摘要生成失败" in (compacted[0].reason or "")
    # History untouched: no summary message was inserted.
    assert not any(
        m.content and m.content.startswith(f"<{SUMMARY_TAG}>")
        for m in agent.messages
    )
    # Guard: exactly one summary attempt this turn despite two loop
    # iterations both being over threshold.
    assert client.summary_attempts == 1
    # Main flow continued anyway.
    assert "final" in "".join(e.text for e in events if isinstance(e, TextDelta))

    # The guard resets on the next user turn (and fails again here).
    await _collect(agent.run("again"))
    assert client.summary_attempts == 2


@pytest.mark.asyncio
async def test_manual_compact_skips_threshold(workdir):
    cfg = _compact_cfg()
    fake_llm = FakeLLMClient(
        [[TextChunk(text="SUMMARY"), CompletionMeta()]],
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    # History too short: polite no-op, no LLM call.
    events = await _collect(agent.compact(trigger="manual"))
    assert not events[0].compacted
    assert "无需压缩" in (events[0].reason or "")
    assert fake_llm.calls == []

    # Enough history: compacts even though usage never crossed the threshold.
    agent.messages.append(Message(role="user", content="x" * 3000))
    agent.messages.append(Message(role="assistant", content="y" * 3000))
    agent.messages.append(Message(role="user", content="z" * 3000))
    agent.messages.append(Message(role="user", content="latest"))
    events = await _collect(agent.compact(trigger="manual"))
    assert events[0].compacted and events[0].trigger == "manual"
    assert agent.messages[1].content.startswith(f"<{SUMMARY_TAG}>")


@pytest.mark.asyncio
async def test_compaction_freezes_title_before_summarizing(workdir):
    """Compaction inside the very first turn must not let the summary
    become the session title (titles are re-derived on every save)."""
    cfg = _compact_cfg()
    # A big tool result from round one lands *before* the recent tail, so
    # find_split_point has something to summarize within the first turn.
    (workdir / "big.txt").write_text("q" * 5000)
    fake_llm = FakeLLMClient(
        [
            [
                ToolCallEvent(id="c1", name="read", arguments={"path": "big.txt"}),
                CompletionMeta(usage={"prompt_tokens": 100}),
            ],
            [
                ToolCallEvent(id="c2", name="ls", arguments={}),
                _big_usage(),
            ],
            [TextChunk(text="SUMMARY"), CompletionMeta()],
            [TextChunk(text="final"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    events = await _collect(agent.run("fix the frobnicate bug please"))

    assert any(
        isinstance(e, CompactionEvent) and e.compacted for e in events
    ), "expected compaction within the first turn"
    # The title comes from the real first user message, not the summary.
    assert agent.session_meta.title.startswith("fix the frobnicate bug")


@pytest.mark.asyncio
async def test_resume_restores_iterative_summary_chain(workdir):
    """P2-3: after resume, the next compaction folds into the old summary."""
    cfg = _compact_cfg()
    session_dir = workdir / "sessions"
    fake_llm = FakeLLMClient(
        [
            [
                TextChunk(text="working"),
                ToolCallEvent(id="c1", name="ls", arguments={}),
                _big_usage(),
            ],
            [TextChunk(text="FIRST SUMMARY"), CompletionMeta()],
            [TextChunk(text="final"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=session_dir,
    )
    _seed_history(agent)
    await _collect(agent.run("x" * 2000))

    resumed_llm = FakeLLMClient(
        [
            [
                TextChunk(text="more work"),
                ToolCallEvent(id="c2", name="ls", arguments={}),
                _big_usage(),
            ],
            [TextChunk(text="SECOND SUMMARY"), CompletionMeta()],
            [TextChunk(text="done"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    resumed = Agent(
        config=cfg,
        llm_client=resumed_llm,
        workdir=workdir,
        session_dir=session_dir,
        resume=agent._session_file,
    )
    assert resumed._previous_summary == "FIRST SUMMARY"
    _seed_history(resumed)  # grow the history past keep_recent again

    await _collect(resumed.run("carry on"))

    # The second summary request received the first summary as context.
    summary_request = resumed_llm.calls[1][0]
    assert "<previous-summary>" in summary_request[0].content
    assert "FIRST SUMMARY" in summary_request[0].content
    assert resumed._previous_summary == "SECOND SUMMARY"


@pytest.mark.asyncio
async def test_auto_compaction_noop_reports_once_per_turn(workdir):
    """P3-1: tail message too big to split -> auto path retries every
    iteration but reports the no-op at most once per turn."""
    cfg = _compact_cfg()
    (workdir / "big.txt").write_text("q" * 5000)  # tool result > keep_recent
    fake_llm = FakeLLMClient(
        [
            [
                ToolCallEvent(id="c1", name="read", arguments={"path": "big.txt"}),
                _big_usage(),
            ],
            [
                ToolCallEvent(id="c2", name="read", arguments={"path": "big.txt"}),
                _big_usage(),
            ],
            [TextChunk(text="final"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))

    compacted = [e for e in events if isinstance(e, CompactionEvent)]
    # Two over-threshold iterations, each with an unsplittable tail, but
    # only one no-op report — and the wording is no longer "历史太短".
    assert len(compacted) == 1
    assert not compacted[0].compacted
    assert "无法安全切分" in (compacted[0].reason or "")
    # No summary call was attempted; the turn completed normally.
    assert len(fake_llm.calls) == 3
    assert "final" in "".join(e.text for e in events if isinstance(e, TextDelta))


@pytest.mark.asyncio
async def test_auto_compaction_crash_does_not_kill_turn(workdir, monkeypatch):
    """P3-3: an unexpected compaction error is contained, reported once,
    and the turn continues."""
    import limbo.agent as agent_module

    def boom(messages, keep_recent_tokens):
        raise RuntimeError("split exploded")

    monkeypatch.setattr(agent_module, "find_split_point", boom)
    cfg = _compact_cfg()
    fake_llm = FakeLLMClient(
        [
            [
                TextChunk(text="working"),
                ToolCallEvent(id="c1", name="ls", arguments={}),
                _big_usage(),
            ],
            [TextChunk(text="final"), CompletionMeta(usage={"prompt_tokens": 100})],
        ]
    )
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )

    events = await _collect(agent.run("hi"))

    compacted = [e for e in events if isinstance(e, CompactionEvent)]
    assert len(compacted) == 1
    assert not compacted[0].compacted
    assert "自动压缩异常" in (compacted[0].reason or "")
    assert agent._auto_compaction_failed_this_turn  # guard engaged
    assert "final" in "".join(e.text for e in events if isinstance(e, TextDelta))


def test_compaction_config_cross_checked_against_window(workdir, monkeypatch):
    """P3-2: reserve + keep_recent must fit inside the context window."""
    import limbo.agent as agent_module
    from limbo.llm.catalog import GENERIC_OPENAI, ModelSpec

    def tiny_window(model_id: str) -> ModelSpec:
        return ModelSpec(id=model_id, provider=GENERIC_OPENAI, context_window=30_000)

    monkeypatch.setattr(agent_module, "resolve_model", tiny_window)
    cfg = Config()  # defaults: 16384 + 20000 = 36384 >= 30000
    with pytest.warns(UserWarning) as record:
        agent = Agent(
            config=cfg,
            llm_client=FakeLLMClient([]),
            workdir=workdir,
            session_dir=workdir / "sessions",
        )
    messages = [str(w.message) for w in record]
    assert any("reset to defaults" in m for m in messages)
    assert any("too small for auto-compaction" in m for m in messages)
    assert not agent._compaction_config.enabled

    def roomy_window(model_id: str) -> ModelSpec:
        return ModelSpec(id=model_id, provider=GENERIC_OPENAI, context_window=40_000)

    monkeypatch.setattr(agent_module, "resolve_model", roomy_window)
    agent = Agent(
        config=cfg,
        llm_client=FakeLLMClient([]),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    assert agent._compaction_config.enabled
    assert agent._compaction_config.reserve_tokens == 16_384


def _make_agent(workdir, model: str):
    cfg = Config()
    cfg.llm.model = model
    fake_llm = FakeLLMClient([[TextChunk(text="ok")]])
    return Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )


def _last_user_message(agent) -> Message:
    return next(m for m in reversed(agent.messages) if m.role == "user")


@pytest.mark.asyncio
async def test_image_attachment_sent_as_blocks_for_vision_model(workdir, tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    agent = _make_agent(workdir, "kimi-k2.5")
    attachment = Attachment(
        kind="image", name="shot.png", path=str(image), mime="image/png"
    )
    await _collect(agent.run("看图 [图片 #1]", attachments=[attachment]))
    user_msg = _last_user_message(agent)
    assert user_msg.content == "看图 [图片 #1]"
    assert user_msg.attachments == [attachment]


@pytest.mark.asyncio
async def test_image_attachment_degrades_for_non_vision_model(workdir, tmp_path):
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    agent = _make_agent(workdir, "deepseek-chat")
    attachment = Attachment(
        kind="image", name="shot.png", path=str(image), mime="image/png"
    )
    await _collect(agent.run("看图 [图片 #1]", attachments=[attachment]))
    user_msg = _last_user_message(agent)
    # No image blocks for non-vision models; a path reference is appended
    # instead so the paste is never silently dropped.
    assert user_msg.attachments is None
    assert str(image) in user_msg.content
    assert "不支持图像输入" in user_msg.content


@pytest.mark.asyncio
async def test_missing_image_noted_but_not_dropped(workdir, tmp_path):
    agent = _make_agent(workdir, "kimi-k2.5")
    attachment = Attachment(
        kind="image", name="gone.png", path=str(tmp_path / "gone.png")
    )
    await _collect(agent.run("看图", attachments=[attachment]))
    user_msg = _last_user_message(agent)
    assert user_msg.attachments is None
    assert "文件已不存在" in user_msg.content


@pytest.mark.asyncio
async def test_small_text_file_attachment_inlined(workdir, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("remember the milk")
    agent = _make_agent(workdir, "deepseek-chat")
    attachment = Attachment(kind="file", name="note.txt", path=str(note))
    await _collect(agent.run("总结这个文件", attachments=[attachment]))
    user_msg = _last_user_message(agent)
    assert "remember the milk" in user_msg.content
    assert user_msg.attachments is None


@pytest.mark.asyncio
async def test_large_or_binary_file_referenced_by_path(workdir, tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"\x00" * (ATTACHMENT_INLINE_MAX_BYTES + 1))
    agent = _make_agent(workdir, "deepseek-chat")
    attachment = Attachment(kind="file", name="big.bin", path=str(big))
    await _collect(agent.run("分析这个文件", attachments=[attachment]))
    user_msg = _last_user_message(agent)
    assert str(big) in user_msg.content
    assert "read 工具" in user_msg.content


@pytest.mark.asyncio
async def test_agent_persists_and_restores_allowed_roots(workdir):
    """Grants made mid-session are saved to the meta and restored on resume."""
    from limbo.sessions import latest_session

    session_dir = workdir / "sessions"
    cfg = Config()
    agent1 = Agent(
        config=cfg,
        llm_client=FakeLLMClient([[TextChunk(text="one")]]),
        workdir=workdir,
        session_dir=session_dir,
    )
    granted = workdir.parent / f"{workdir.name}-granted"
    granted.mkdir(exist_ok=True)
    agent1.registry.add_allowed_roots([granted])
    await _collect(agent1.run("look outside"))

    agent2 = Agent(
        config=cfg,
        llm_client=FakeLLMClient([[TextChunk(text="two")]]),
        workdir=workdir,
        session_dir=session_dir,
        resume=latest_session(session_dir),
    )
    assert granted.resolve() in agent2.registry.allowed_roots
    # The restored fence actually lets tools through.
    ls_tool = agent2.registry.get("ls")
    assert ls_tool is not None
    assert ls_tool.is_within_scope(granted.resolve())


# -- usage counter normalization ----------------------------------------------


def test_extract_cached_tokens_responses_dialect():
    # The Responses API nests cache hits under input_tokens_details.
    usage = {
        "input_tokens": 164,
        "output_tokens": 20,
        "total_tokens": 184,
        "input_tokens_details": {"cached_tokens": 64},
    }
    assert _extract_cached_tokens(usage) == 64


def test_extract_cached_tokens_existing_branches_win_first():
    assert _extract_cached_tokens({"prompt_cache_hit_tokens": 10}) == 10
    assert (
        _extract_cached_tokens({"prompt_tokens_details": {"cached_tokens": 20}})
        == 20
    )
    assert _extract_cached_tokens({"cache_read_input_tokens": 30}) == 30
    assert _extract_cached_tokens({"input_tokens": 5}) is None
    assert _extract_cached_tokens(None) is None
