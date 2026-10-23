"""Tests for the Agent conversation loop."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from limbo.agent import (
    Agent,
    CompactionEvent,
    ErrorEvent,
    SteerEvent,
    TextDelta,
    ToolCallRequest,
    ToolResultEvent,
    UsageUpdate,
)
from limbo.compaction import SUMMARY_TAG
from limbo.config import CompactionSettings, Config, LLMConfig
from limbo.llm.retry import LLMHttpError
from limbo.models import (
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
            if isinstance(event, BaseException):
                raise event  # a stream that breaks mid-response
            yield event


async def _collect(run):
    return [event async for event in run]


def read_agent_trace(agent: Agent) -> list[dict]:
    """read_trace with a flush barrier: log() only enqueues (the background
    writer thread does the disk write), so reading right after a turn can
    race the writer under suite load."""
    agent.trace.flush()
    return read_trace(agent.trace.path)


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

    async def crashing_run(call, **_):
        raise RuntimeError("tool exploded")

    agent.registry.run = crashing_run

    events = await _collect(agent.run("read main.py"))
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.success is False
    assert "tool exploded" in (result_events[0].result.error or "")
    # The loop continued and produced the final text response.
    assert any(isinstance(e, TextDelta) and e.text == "done" for e in events)
    assert not any(isinstance(e, ErrorEvent) for e in events)


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
    original_run = agent.registry.run

    async def tracking_run(call, **kwargs):
        executed.append((call.name, call.arguments))
        return await original_run(call, **kwargs)

    agent.registry.run = tracking_run

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


# --- session resume -------------------------------------------------------


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
async def test_agent_resume_reruns_safe_calls_and_flags_unsafe_ones(workdir):
    from limbo.history import NOT_STARTED_CONTENT, PARTIAL_CONTENT
    from limbo.journal import SessionJournal
    from limbo.sessions import SessionMeta, load_session_full, save_session

    (workdir / "notes.txt").write_text("hello from disk\n")
    session_dir = workdir / "sessions"
    session_dir.mkdir()
    path = session_dir / "crashed.jsonl"

    def call(call_id: str, name: str, arguments: dict) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }

    save_session(
        path,
        SessionMeta(id="crashed", tool_intents=True),
        [
            Message(role="user", content="look and change"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    call("r1", "read", {"path": "notes.txt"}),
                    call("b1", "bash", {"command": "make deploy"}),
                    call("w1", "write", {"path": "x.txt", "content": "x"}),
                ],
            ),
        ],
    )
    # The process died after read and bash started, before write did.
    SessionJournal(path).record_intents([("r1", "read", "safe"), ("b1", "bash", "unsafe")])

    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([[TextChunk(text="ok")]]),
        workdir=workdir,
        session_dir=session_dir,
        resume=path,
    )
    events = await _collect(agent.run("continue"))

    reruns = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [e.id for e in reruns] == ["r1"]
    sent = agent.llm_client.calls[0][0]
    contents = {m.tool_call_id: m.content for m in sent if m.role == "tool"}
    assert "hello from disk" in (contents["r1"] or "")
    assert contents["b1"] == PARTIAL_CONTENT
    assert contents["w1"] == NOT_STARTED_CONTENT
    # The unsafe calls never ran again.
    assert not (workdir / "x.txt").exists()
    # The re-run result is what the session file now holds.
    on_disk = load_session_full(path)
    disk_contents = {
        m.tool_call_id: m.content for m in on_disk.messages if m.role == "tool"
    }
    assert "hello from disk" in (disk_contents["r1"] or "")
    trace = read_agent_trace(agent)
    assert any(
        r["type"] == "tool_recovery" and r["rerun_tool_call_ids"] == ["r1"]
        for r in trace
    )


def _resume(agent: Agent, responses=()) -> Agent:
    return Agent(
        config=Config(),
        llm_client=FakeLLMClient(list(responses)),
        workdir=agent.workdir,
        session_dir=agent.session_path.parent,
        resume=agent.session_path,
    )


@pytest.mark.asyncio
async def test_queued_messages_survive_a_crash(workdir):
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient([[TextChunk(text="a")]]),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("hi"))
    # Queued while a task runs; the process dies before they are drained.
    steer_id = agent.steer("also check the docs")
    follow_id = agent.steer("then write a summary", mode="follow_up")

    resumed = _resume(agent, [[TextChunk(text="b")]])

    assert [(i.id, i.text, i.mode) for i in resumed.queued_items] == [
        (steer_id, "also check the docs", "steer"),
        (follow_id, "then write a summary", "follow_up"),
    ]
    # The next turn injects the steer before its input; the follow-up
    # stays for the pump's end-of-task drain.
    events = await _collect(resumed.run("continue"))
    assert [e.id for e in events if isinstance(e, SteerEvent)] == [steer_id]
    assert [i.id for i in resumed.queued_items] == [follow_id]
    # The drained steer is journaled as dequeued; the follow-up is not.
    again = _resume(resumed)
    assert [i.id for i in again.queued_items] == [follow_id]


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

    records = read_agent_trace(agent)
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
    assert start["config"]["model"] == "deepseek-v4-pro"
    assert start["resumed"] is False
    assert start["trace_schema"] == 2
    # The API key must never appear in the trace.
    assert "api_key" not in json.dumps(records)

    assert records[1]["content"] == "read main.py"

    request = records[2]
    assert request["turn"] == 1 and request["iteration"] == 1
    assert request["attempt"] == 1
    assert request["body"]["model"] == "fake-model"
    assert request["body"]["messages"][0]["role"] == "system"
    assert request["body"]["tools"]  # full tool definitions included

    response = records[3]
    assert response["usage"] == usage
    assert response["cached_tokens"] == 80
    assert response["finish_reason"] == "tool_calls"
    assert response["tool_calls"] == [
        {"id": "c1", "name": "read", "arguments": {"path": "main.py"}}
    ]
    assert response["content"] == ""
    assert response["started_at"] <= response["ts"]
    assert records[7]["content"] == "done"

    tool_call, tool_result = records[4], records[5]
    assert tool_call["name"] == "read"
    assert tool_call["arguments"] == {"path": "main.py"}
    assert tool_result["success"] is True
    assert tool_result["duration"] >= 0
    assert tool_result["started_at"] <= tool_result["ts"]
    assert "x = 1" in tool_result["output"]

    turn_end = records[-1]
    assert turn_end["status"] == "completed"
    assert turn_end["started_at"] <= records[1]["ts"]
    assert turn_end["iterations"] == 2


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

    async def slow_run(call, **_):
        await asyncio.sleep(0.3)
        return ToolResult(success=True, output=f"ok:{call.name}")

    agent.registry.run = slow_run

    start = time.monotonic()
    events = await _collect(agent.run("read both"))
    elapsed = time.monotonic() - start

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(e.result.success for e in result_events)
    # Serial execution would take ~0.6s; concurrent should be well under.
    assert elapsed < 0.55


# -- auto-compaction (LIM-14) ---------------------------------------------------


def _compact_cfg(**overrides) -> Config:
    # 1000 is the validator floor; tests pad messages so history exceeds it.
    defaults = {"keep_recent_tokens": 1000}
    defaults.update(overrides)
    # Lock a small window (generic fallback = 128K) so these tests don't
    # depend on the default model's context size — deepseek-v4-pro grew to
    # 1M, which would make the 120K prompt fixtures fall below the
    # compaction threshold.
    return Config(
        compaction=CompactionSettings(**defaults),
        llm=LLMConfig(model="my-local-model"),
    )


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


class _OverflowingClient:
    """Rejects the main request with a context-overflow error ``overflows``
    times; summary requests (no tools) succeed."""

    def __init__(self, overflows: int):
        self.calls = []
        self.overflows = overflows

    async def chat(self, messages, tools, on_request=None):
        self.calls.append((messages, tools))
        if not tools:
            yield TextChunk(text="SUMMARY")
            yield CompletionMeta()
            return
        if self.overflows:
            self.overflows -= 1
            raise LLMHttpError(400, "Error code: 400 - prompt is too long: 250000 tokens")
        yield TextChunk(text="final")
        yield CompletionMeta(usage={"prompt_tokens": 100})


@pytest.mark.asyncio
async def test_context_overflow_compacts_and_retries_once(workdir):
    client = _OverflowingClient(overflows=1)
    agent = Agent(
        config=_compact_cfg(),
        llm_client=client,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    _seed_history(agent)

    events = await _collect(agent.run("x" * 2000))

    compacted = [e for e in events if isinstance(e, CompactionEvent)]
    assert [(e.compacted, e.trigger) for e in compacted] == [(True, "overflow")]
    assert not [e for e in events if isinstance(e, ErrorEvent)]
    assert "final" in "".join(e.text for e in events if isinstance(e, TextDelta))
    # Rejected request, summary, then the resend with the compacted history.
    assert [bool(tools) for _, tools in client.calls] == [True, False, True]
    assert client.calls[2][0][1].content.startswith(f"<{SUMMARY_TAG}>")
    trace = read_agent_trace(agent)
    assert any(
        r["type"] == "llm_error" and r["context_overflow"] for r in trace
    )


@pytest.mark.asyncio
async def test_two_appended_compactions_then_resume(workdir):
    from limbo.compaction import is_summary_message

    cfg = _compact_cfg()
    fake_llm = FakeLLMClient(
        [
            [TextChunk(text="ok1")],
            [TextChunk(text="FIRST SUMMARY")],
            [TextChunk(text="c" * 3000)],
            [TextChunk(text="SECOND SUMMARY")],
        ]
    )
    session_dir = workdir / "sessions"
    agent = Agent(
        config=cfg, llm_client=fake_llm, workdir=workdir, session_dir=session_dir
    )
    agent.messages.append(Message(role="user", content="u" * 3000))
    agent.messages.append(Message(role="assistant", content="v" * 3000))
    await _collect(agent.run("w" * 3000))
    await _collect(agent.compact(trigger="manual"))
    await _collect(agent.run("x" * 3000))
    await _collect(agent.compact(trigger="manual"))
    await agent._save_session()
    expected = [m.content for m in agent.messages[2:]]
    # Both compactions were appended to the journal, not rewritten into it.
    raw = agent.session_path.read_text()
    assert raw.count('"type": "compaction"') == 2
    assert "u" * 3000 in raw

    resumed = Agent(
        config=cfg,
        llm_client=FakeLLMClient([]),
        workdir=workdir,
        session_dir=session_dir,
        resume=agent.session_path,
    )

    assert is_summary_message(resumed.messages[1])
    assert "SECOND SUMMARY" in (resumed.messages[1].content or "")
    assert [m.content for m in resumed.messages[2:]] == expected
    # A clean resume keeps the journal: no rewrite is scheduled.
    assert resumed._journal.rewrite_pending is False


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


# -- steer queue (RFC LIM-20) --------------------------------------------------


def _make_steer_agent(workdir, client, max_iterations: int | None = None) -> Agent:
    cfg = Config()
    if max_iterations is not None:
        cfg.llm.max_iterations = max_iterations
    agent = Agent(
        config=cfg,
        llm_client=FakeLLMClient([]),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    agent.llm_client = client
    return agent


@pytest.mark.asyncio
async def test_steer_injected_at_loop_top(workdir):
    """A message steered mid-turn is injected after the tool results and
    before the next LLM request, as a plain user message."""
    (workdir / "main.py").write_text("x = 1\n")

    class SteeringClient:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools, on_request=None):
            self.calls.append([m.model_dump() for m in messages])
            if len(self.calls) == 1:
                yield ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})
                agent.steer("改用 uv")
            else:
                yield TextChunk(text="done")

    client = SteeringClient()
    agent = _make_steer_agent(workdir, client)

    events = await _collect(agent.run("read main.py"))

    # Second request sees: system, user, assistant(tool_calls), tool, user(steer).
    second = client.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "user"]
    assert second[-1]["content"] == "改用 uv"
    # The SteerEvent is emitted before the follow-up response streams.
    kinds = [type(e).__name__ for e in events]
    assert kinds.index("SteerEvent") < kinds.index("TextDelta")
    steer_events = [e for e in events if isinstance(e, SteerEvent)]
    assert len(steer_events) == 1
    assert steer_events[0].text == "改用 uv"
    assert not agent.has_pending_steer()
    # Trace marks the injected message as a steer (the turn's first input has
    # no steer/followup flag).
    records = read_agent_trace(agent)
    steered = [r for r in records if r["type"] == "user_message" and r.get("steer")]
    assert len(steered) == 1 and steered[0]["content"] == "改用 uv"
    first = [
        r
        for r in records
        if r["type"] == "user_message" and not r.get("steer") and not r.get("followup")
    ]
    assert len(first) == 1
    # And the steer message persists with the session.
    from limbo.sessions import load_session

    _, persisted, _ = load_session(agent._session_file)
    assert any(m.role == "user" and m.content == "改用 uv" for m in persisted)


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 32


# -- Agent.close() releases the trace file handle ----------------------------


# --- durable progress: mid-turn checkpoints ------------------------------


@pytest.mark.asyncio
async def test_tool_calls_are_on_disk_before_tools_run(workdir):
    seen_on_disk: list[list[str]] = []
    seen_intents: list[tuple[str, str]] = []

    class SpyTool:
        name = "spy"
        description = "records what the session file holds"
        parameters = {"type": "object", "properties": {}}

        def execute(self, arguments):
            from limbo.sessions import load_session_full

            loaded = load_session_full(agent.session_path)
            seen_on_disk.append([m.role for m in loaded.messages])
            seen_intents.extend((i.call_id, i.replay) for i in loaded.intents)
            return ToolResult(success=True, output="ok")

        def cancel(self):
            pass

    fake_llm = FakeLLMClient(
        [
            [ToolCallEvent(id="c1", name="spy", arguments={})],
            [TextChunk(text="done")],
        ]
    )
    agent = Agent(
        config=Config(),
        llm_client=fake_llm,
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    agent.registry._tools["spy"] = SpyTool()

    await _collect(agent.run("go"))

    assert seen_on_disk == [["system", "user", "assistant"]]
    # The intent is committed before the effect (unknown policy: unsafe).
    assert seen_intents == [("c1", "unsafe")]


# -- usage ledger (T13) ---------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_ledger_accumulates_and_survives_resume(workdir):
    from limbo.llm.catalog import resolve_model

    first = {
        "prompt_tokens": 80,
        "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 50},
    }
    second = {"prompt_tokens": 150, "total_tokens": 200}
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient(
            [
                [TextChunk(text="a"), CompletionMeta(usage=first)],
                [TextChunk(text="b"), CompletionMeta(usage=second)],
            ]
        ),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    events = await _collect(agent.run("one"))
    assert [e.total_tokens for e in events if isinstance(e, UsageUpdate)] == [100]
    await _collect(agent.run("two"))
    assert agent.session_total_tokens == 300
    agent.record_usage(None)  # nothing reported: nothing counted
    assert agent.session_total_tokens == 300

    model = agent.config.llm.model
    key = f"{resolve_model(model).provider.id}/{model}"
    entry = agent.usage_ledger[key]
    assert (entry.requests, entry.prompt_tokens, entry.cached_tokens) == (2, 230, 50)

    resumed = _resume(agent)
    assert resumed.session_total_tokens == 300
    assert resumed.usage_ledger[key].requests == 2


# -- mid-stream retry (T15) -----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [1, 0])
async def test_stream_broken_mid_response_is_resent(workdir, max_retries):
    import httpx

    from limbo.agent import RetryEvent

    cfg = Config()
    cfg.llm.max_retries = max_retries
    cfg.llm.retry_base_delay = 0
    fake = FakeLLMClient(
        [
            [TextChunk(text="half an ans"), httpx.ReadError("connection reset")],
            [TextChunk(text="the full answer")],
        ]
    )
    agent = Agent(
        config=cfg, llm_client=fake, workdir=workdir, session_dir=workdir / "sessions"
    )
    events = await _collect(agent.run("q"))

    retries = [e for e in events if isinstance(e, RetryEvent)]
    if max_retries:
        assert [r.attempt for r in retries] == [1]
        assert len(fake.calls) == 2
        # Only the retried response is kept.
        assert agent.messages[-1].content == "the full answer"
        assert not any("half an ans" in (m.content or "") for m in agent.messages)
    else:
        assert retries == []
        assert any(isinstance(e, ErrorEvent) for e in events)


# -- /fork and /handoff (T16) ---------------------------------------------------


@pytest.mark.asyncio
async def test_fork_and_handoff_leave_the_parent_untouched(workdir):
    agent = Agent(
        config=Config(),
        llm_client=FakeLLMClient(
            [[TextChunk(text="a1")], [TextChunk(text="a2")], [TextChunk(text="c1")]]
        ),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    await _collect(agent.run("first"))
    await _collect(agent.run("second"))
    parent_raw = agent.session_path.read_text()

    # Keep only the first turn; the child loads on its own.
    child = _resume_at(agent, agent.fork(keep_turns=1))
    assert [m.content for m in child.messages[1:]] == ["first", "a1"]
    meta = child.session_meta
    assert meta.parent_session == agent.session_id
    assert meta.fork_at == child.messages[-1].id
    assert meta.title.endswith("(fork)")
    # Work in the child never reaches the parent.
    child.llm_client = agent.llm_client
    await _collect(child.run("child turn"))
    assert agent.session_path.read_text() == parent_raw

    handed = _resume_at(agent, agent.handoff("continue the refactor"))
    assert len(handed.messages) == 1  # system only: a clean context
    assert handed.session_meta.parent_session == agent.session_id
    assert handed.session_meta.fork_at is None
    assert str(agent.session_path) in agent.handoff_prompt("x")


def _resume_at(agent: Agent, path: Path) -> Agent:
    return Agent(
        config=Config(),
        llm_client=FakeLLMClient([]),
        workdir=agent.workdir,
        session_dir=path.parent,
        resume=path,
    )
