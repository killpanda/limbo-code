"""Agent-level tests for Code Mode: toggle, prompt rebuild, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.agent import Agent
from limbo.code_mode import CODE_ONLY_INSTRUCTION
from limbo.config import Config
from limbo.models import CompletionMeta, TextChunk, ToolCallEvent
from limbo.sessions import load_session


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


def _make_agent(workdir: Path, responses) -> Agent:
    return Agent(
        config=Config(),
        llm_client=FakeLLMClient(responses),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )


def test_set_code_mode_rebuilds_system_prompt(tmp_path):
    agent = _make_agent(tmp_path, [])
    assert agent.code_mode is False
    assert "run_code" not in (agent.messages[0].content or "")

    assert agent.set_code_mode(True) is True
    prompt = agent.messages[0].content or ""
    assert CODE_ONLY_INSTRUCTION in prompt
    assert "## Writing code for run_code" in prompt
    # Toggling to the same state is a no-op.
    assert agent.set_code_mode(True) is False

    assert agent.set_code_mode(False) is True
    prompt = agent.messages[0].content or ""
    assert CODE_ONLY_INSTRUCTION not in prompt
    assert "run_code" not in prompt


def test_code_mode_collapses_wire_tools(tmp_path):
    agent = _make_agent(tmp_path, [])
    agent.set_code_mode(True)
    defs = agent.registry.definitions()
    assert [d["function"]["name"] for d in defs] == ["run_code"]


@pytest.mark.asyncio
async def test_code_mode_executes_run_code_turn(tmp_path):
    (tmp_path / "note.txt").write_text("code mode e2e")
    # The model writes a program; the agent executes it as one tool call.
    fake = FakeLLMClient(
        [
            [
                ToolCallEvent(
                    id="call-1",
                    name="run_code",
                    arguments={
                        "code": (
                            'content = await tools.read(path="note.txt")\n'
                            'return f"read: {content}"'
                        ),
                        "description": "Read note.txt",
                    },
                ),
                CompletionMeta(finish_reason="tool_calls", usage=None),
            ],
            [TextChunk(text="done"), CompletionMeta(finish_reason="stop")],
        ]
    )
    agent = Agent(
        config=Config(),
        llm_client=fake,
        workdir=tmp_path,
        session_dir=tmp_path / "sessions",
    )
    agent.set_code_mode(True)
    # The request carried exactly one tool: run_code.
    await _collect(agent.run("use code mode"))
    assert [d["function"]["name"] for d in fake.calls[0][1]] == ["run_code"]
    # The tool result entered history (curated program output only).
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "read: code mode e2e" in (tool_msgs[0].content or "")


def test_code_mode_persisted_and_restored_on_resume(tmp_path):
    agent = _make_agent(tmp_path, [])
    agent.set_code_mode(True)
    agent._save_session_sync()

    meta, history, _ = load_session(agent.session_path)
    assert meta.code_mode is True

    # Resume: a fresh agent on the same file restores code mode and its
    # system prompt reflects the SDK.
    resumed = Agent(
        config=Config(),
        llm_client=FakeLLMClient([]),
        workdir=tmp_path,
        session_dir=tmp_path / "sessions",
        resume=agent.session_path,
    )
    assert resumed.code_mode is True
    prompt = resumed.messages[0].content or ""
    assert CODE_ONLY_INSTRUCTION in prompt


def test_native_mode_not_persisted(tmp_path):
    agent = _make_agent(tmp_path, [])
    assert agent.code_mode is False
    agent._save_session_sync()
    meta, _, _ = load_session(agent.session_path)
    assert meta.code_mode is False
