"""Agent-level tests for plan mode: toggle, prompt, enforcement, persistence."""

from __future__ import annotations

import pytest

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import CompletionMeta, TextChunk, ToolCallEvent


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools, on_request=None):
        for event in self.responses.pop(0):
            yield event


async def _collect(run):
    return [event async for event in run]


def _make_agent(workdir, responses) -> Agent:
    return Agent(
        config=Config(),
        llm_client=FakeLLMClient(responses),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )


@pytest.mark.asyncio
async def test_plan_mode_blocks_a_write_call_in_a_turn(tmp_path):
    agent = _make_agent(
        tmp_path,
        [
            [
                ToolCallEvent(
                    id="c1", name="write", arguments={"path": "x.txt", "content": "a"}
                ),
                CompletionMeta(),
            ],
            [TextChunk(text="here is the plan"), CompletionMeta()],
        ],
    )
    agent.set_plan_mode(True)
    await _collect(agent.run("change x"))

    assert not (tmp_path / "x.txt").exists()
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Plan mode" in (tool_msgs[0].content or "")




@pytest.mark.asyncio
async def test_plan_mode_blocks_run_code_in_a_turn(tmp_path):
    """Code Mode's transport runs arbitrary Python: plan mode must refuse it."""
    agent = _make_agent(
        tmp_path,
        [
            [
                ToolCallEvent(
                    id="c1",
                    name="run_code",
                    arguments={
                        "code": "open('x.txt', 'w').write('a')",
                        "description": "write x",
                    },
                ),
                CompletionMeta(),
            ],
            [TextChunk(text="here is the plan"), CompletionMeta()],
        ],
    )
    agent.set_plan_mode(True)
    await _collect(agent.run("change x"))

    assert not (tmp_path / "x.txt").exists()
    tool_msgs = [m for m in agent.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "Plan mode" in (tool_msgs[0].content or "")
