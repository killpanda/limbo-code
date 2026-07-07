import pytest

from limbo.agent import Agent
from limbo.config import Config
from limbo.models import TextChunk, ToolCallEvent


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses

    async def chat(self, messages, tools):
        for event in self.responses.pop(0):
            yield event


async def _collect(run):
    return [event async for event in run]


@pytest.mark.asyncio
async def test_integration_read_and_answer(tmp_path):
    (tmp_path / "main.py").write_text("x = 1\n")
    cfg = Config()
    fake_llm = FakeLLM([
        [ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})],
        [TextChunk(text="The file sets x = 1.")],
    ])
    agent = Agent(
        config=cfg,
        llm_client=fake_llm,
        workdir=tmp_path,
        session_dir=tmp_path / "sessions",
    )
    events = await _collect(agent.run("what is in main.py"))
    assert any(hasattr(e, "name") and e.name == "read" for e in events)
    assert any(hasattr(e, "text") and "x = 1" in e.text for e in events)
