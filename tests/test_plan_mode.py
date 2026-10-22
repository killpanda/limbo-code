"""Agent-level tests for plan mode: toggle, prompt, enforcement, persistence."""

from __future__ import annotations

import pytest

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import CompletionMeta, TextChunk, ToolCallEvent
from limbo.sessions import load_session
from limbo.toggles import parse_toggle
from limbo.tools.hooks import PLAN_MODE_INSTRUCTION


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


def test_set_plan_mode_rebuilds_system_prompt(tmp_path):
    agent = _make_agent(tmp_path, [])
    assert agent.plan_mode is False
    assert PLAN_MODE_INSTRUCTION not in (agent.messages[0].content or "")

    assert agent.set_plan_mode(True) is True
    assert PLAN_MODE_INSTRUCTION in (agent.messages[0].content or "")
    assert agent.set_plan_mode(True) is False

    assert agent.set_plan_mode(False) is True
    assert PLAN_MODE_INSTRUCTION not in (agent.messages[0].content or "")


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


def test_plan_mode_persisted_and_restored_on_resume(tmp_path):
    agent = _make_agent(tmp_path, [])
    agent.set_plan_mode(True)
    agent._save_session_sync()
    meta, _, _ = load_session(agent.session_path)
    assert meta.plan_mode is True

    resumed = Agent(
        config=Config(),
        llm_client=FakeLLMClient([]),
        workdir=tmp_path,
        session_dir=tmp_path / "sessions",
        resume=agent.session_path,
    )
    assert resumed.plan_mode is True
    assert resumed.code_mode is False
    assert PLAN_MODE_INSTRUCTION in (resumed.messages[0].content or "")


def test_parse_toggle():
    assert parse_toggle("on", False) is True
    assert parse_toggle("关", True) is False
    assert parse_toggle("", True) is False
    assert parse_toggle(" ", False) is True
    assert parse_toggle("maybe", False) is None
