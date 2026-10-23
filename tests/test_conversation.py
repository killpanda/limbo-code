"""Tests for Conversation: the session lifecycle both frontends share.

Seam under test: the ``Conversation`` verbs, driven with a real Agent +
TurnPump and a scripted fake LLM, plus a recording reporter for the
lifecycle states the integrations see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.config import Config
from limbo.conversation import Conversation, VerifyProposals
from limbo.models import TextChunk


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, messages, tools, on_request=None):
        for event in self.responses.pop(0):
            yield event


class RecordingReporter:
    def __init__(self):
        self.calls: list[tuple] = []

    def report(self, state, *, message=None, session_id=None, session_path=None):
        self.calls.append(("report", state))

    def report_session(self, session_id, session_path=None):
        self.calls.append(("session", session_id))

    def release(self):
        pass

    @property
    def states(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "report"]


def make_conversation(tmp_path: Path, responses, reporter=None) -> Conversation:
    return Conversation(
        config=Config(),
        workdir=tmp_path,
        llm_client=FakeLLMClient(responses),
        session_dir=tmp_path / "sessions",
        reporters=reporter,
    )


async def drain(conversation: Conversation, events) -> None:
    await conversation.drive(events, lambda event: None)


@pytest.mark.asyncio
async def test_failed_switch_keeps_the_current_session_live(tmp_path):
    reporter = RecordingReporter()
    conversation = make_conversation(
        tmp_path, [[TextChunk(text="one")], [TextChunk(text="two")]], reporter
    )
    await drain(conversation, conversation.pump.run("first"))
    before = conversation.agent

    with pytest.raises(OSError):
        conversation.switch(tmp_path / "sessions" / "deleted.jsonl")

    # Nothing was replaced or closed: the same session keeps working.
    assert conversation.agent is before
    assert ("session", before.session_id) not in reporter.calls
    await drain(conversation, conversation.pump.run("second"))
    assert [m.content for m in before.messages if m.role == "user"] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_goal_proposal_round_blocks_then_accepting_starts_round_one(tmp_path):
    reporter = RecordingReporter()
    proposal = (
        "看过了。<verify_proposal><command>pytest -x</command>"
        "<command>make check</command></verify_proposal>"
    )
    conversation = make_conversation(
        tmp_path, [[TextChunk(text=proposal)]], reporter
    )

    prompt = conversation.start_goal("修好测试")
    await drain(conversation, conversation.pump.run(prompt))

    result = conversation.take_verify_proposal()
    assert result == VerifyProposals(["pytest -x", "make check"])
    assert conversation.awaiting_verify
    assert reporter.states == ["working", "done", "blocked"]

    first_round = conversation.accept_verify("make check")
    assert first_round is not None and "修好测试" in first_round
    assert not conversation.awaiting_verify
    state = conversation.pump.status()
    assert state is not None and state.verify_command == "make check"
