"""Agent-level tests for Code Mode: toggle, prompt rebuild, persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import CompletionMeta, TextChunk, ToolCallEvent
from limbo.trace import read_trace


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
    # The sub-call is traced under its run_code parent (never in history).
    agent.trace.flush()
    subs = [
        r for r in read_trace(agent.trace.path)
        if r.get("parent_call_id") == "call-1"
    ]
    assert [(r["type"], r["name"]) for r in subs] == [
        ("tool_call", "read"),
        ("tool_result", "read"),
    ]
    assert subs[1]["success"] is True
    assert subs[0]["id"] == subs[1]["id"]


