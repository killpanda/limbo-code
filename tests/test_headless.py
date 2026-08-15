"""Tests for the headless CLI frontend (RFC LIM-60, design/rfc-headless-cli.md).

Seam under test: ``HeadlessFrontend.run(inputs, out)`` — a real Agent driven
by a scripted FakeLLMClient (same pattern as test_pump.py), a recording fake
reporter for lifecycle assertions, and an async iterator of InputEvents as
stdin. Output is collected into a list of written chunks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import limbo.config as config_module
import limbo.pump as pump_module
from limbo.agent import Agent
from limbo.config import Config
from limbo.goal import VerifyResult
from limbo.headless import (
    AcceptVerify,
    Echo,
    Eof,
    HeadlessFrontend,
    Interrupt,
    InvalidVerifyAnswer,
    Prompt,
    SkipVerify,
    SlashGoal,
    SlashModel,
    SlashNew,
    SlashQuit,
    Submit,
    TerminalLineParser,
    UnknownCommand,
    parse_submission,
    parse_verify_answer,
)
from limbo.models import TextChunk, ToolCallEvent
from limbo.pump import TurnPump


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools, on_request=None):
        self.calls.append([m for m in messages])
        response = self.responses.pop(0)
        if callable(response):
            response = await response()
        for event in response:
            yield event


class RecordingReporter:
    """IntegrationReporter-shaped recorder (release included)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def report(self, state, *, message=None, session_id=None, session_path=None):
        self.calls.append(("report", state, message))

    def report_session(self, session_id, session_path=None):
        self.calls.append(("session", session_id))

    def release(self):
        self.calls.append(("release",))

    @property
    def states(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "report"]


def make_frontend(workdir: Path, responses, reporter=None):
    """Build a frontend wired to a real Agent + scripted LLM."""
    config = Config()
    agent = Agent(
        config=config,
        llm_client=FakeLLMClient(responses),
        workdir=workdir,
        session_dir=workdir / "sessions",
    )
    pump = TurnPump(agent, workdir)
    reporter = reporter or RecordingReporter()
    chunks: list[str] = []

    def make_agent(client):
        return Agent(
            config=config,
            llm_client=client,
            workdir=workdir,
            session_dir=workdir / "sessions",
        )

    frontend = HeadlessFrontend(
        config=config,
        agent=agent,
        pump=pump,
        llm_client=agent.llm_client,
        reporters=reporter,
        out=chunks.append,
        agent_factory=make_agent,
        pump_factory=lambda a: TurnPump(a, workdir),
        echo_input=True,
    )
    return frontend, chunks, reporter


async def feed(events):
    for event in events:
        yield event


def output_of(chunks) -> str:
    return "".join(chunks)


async def wait_until(cond, timeout=2.0):
    async def _poll():
        while not cond():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout)


def gated_response(started: asyncio.Event, release: asyncio.Event, events):
    """An LLM response that blocks until the test releases it."""

    async def gated():
        started.set()
        await release.wait()
        return events

    return gated


# -- single turn ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_echoes_streams_and_reports(workdir):
    frontend, chunks, reporter = make_frontend(
        workdir, [[TextChunk(text="hello world")]]
    )

    rc = await frontend.run(feed([Submit("你好"), Eof()]))

    assert rc == 0
    out = output_of(chunks)
    assert "❯ 你好" in out  # pipe-mode echo of the submission
    assert "hello world" in out  # streamed assistant text
    assert "── limbo idle ·" in out  # turn-end marker
    # Lifecycle: session -> idle -> working -> idle, released at the end.
    assert reporter.states == ["idle", "working", "idle"]
    assert reporter.calls[0][0] == "session"
    assert reporter.calls[-1] == ("release",)


# -- tool rendering ------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_and_result_render(workdir):
    (workdir / "hello.txt").write_text("hi")
    frontend, chunks, _ = make_frontend(
        workdir,
        [
            [ToolCallEvent(id="t1", name="ls", arguments={"path": "."})],
            [TextChunk(text="看到了 hello.txt")],
        ],
    )

    rc = await frontend.run(feed([Submit("列一下目录"), Eof()]))

    assert rc == 0
    out = output_of(chunks)
    assert "⚙ ls(" in out
    assert "✓ ls" in out
    assert "hello.txt" in out  # tool output is rendered for herdr to read
    assert "看到了 hello.txt" in out


# -- steer while busy ----------------------------------------------------------


@pytest.mark.asyncio
async def test_steer_while_busy_queues_and_follows_up(workdir):
    started = asyncio.Event()
    release = asyncio.Event()
    frontend, chunks, _ = make_frontend(
        workdir,
        [
            gated_response(started, release, [TextChunk(text="第一轮完成")]),
            [TextChunk(text="第二轮完成")],
        ],
    )

    async def script():
        yield Submit("第一个任务")
        await started.wait()
        yield Submit("补充要求")
        await wait_until(lambda: "已排队插话" in output_of(chunks))
        release.set()
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    out = output_of(chunks)
    assert "➤ 已排队插话：补充要求" in out
    assert "插话已注入" in out  # SteerEvent at the turn boundary
    assert "第一轮完成" in out
    assert "第二轮完成" in out


# -- interrupt -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_cancels_stream(workdir):
    started = asyncio.Event()

    async def hangs():
        started.set()
        await asyncio.Event().wait()  # never completes; interrupt cancels it
        return []

    frontend, chunks, reporter = make_frontend(workdir, [hangs])

    async def script():
        yield Submit("长跑任务")
        await started.wait()
        yield Interrupt(hard=False)  # Esc
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    assert "⏹ 已打断" in output_of(chunks)
    assert reporter.states == ["idle", "working", "idle"]


@pytest.mark.asyncio
async def test_ctrl_c_quits_from_idle(workdir):
    frontend, _, _ = make_frontend(workdir, [])

    rc = await frontend.run(feed([Interrupt(hard=True)]))

    assert rc == 0  # quit without waiting for more input


# -- commands refused while busy ------------------------------------------------


@pytest.mark.asyncio
async def test_history_commands_refused_while_busy(workdir):
    started = asyncio.Event()
    release = asyncio.Event()
    frontend, chunks, _ = make_frontend(
        workdir, [gated_response(started, release, [TextChunk(text="done")])]
    )

    async def script():
        yield Submit("任务")
        await started.wait()
        yield Submit("/goal 新目标")
        yield Submit("/compact")
        release.set()
        await wait_until(lambda: "── limbo idle ·" in output_of(chunks))
        yield Submit("/goal")  # query: the refused /goal never took effect
        await wait_until(lambda: "当前没有 goal" in output_of(chunks))
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    out = output_of(chunks)
    assert out.count("该命令暂不可用") == 2
    assert "当前没有 goal" in out


# -- EOF while busy -------------------------------------------------------------


@pytest.mark.asyncio
async def test_eof_waits_for_busy_turn(workdir):
    started = asyncio.Event()
    release = asyncio.Event()
    frontend, chunks, _ = make_frontend(
        workdir, [gated_response(started, release, [TextChunk(text="finished")])]
    )

    async def script():
        yield Submit("任务")
        await started.wait()
        yield Eof()
        await wait_until(lambda: "等待当前任务完成" in output_of(chunks))
        release.set()

    rc = await frontend.run(script())

    assert rc == 0
    out = output_of(chunks)
    assert "输入已关闭，等待当前任务完成" in out
    assert "finished" in out  # the turn ran to completion before exit


# -- goal: proposal -> blocked -> confirm ---------------------------------------

PROPOSAL = (
    "我先了解一下仓库。\n"
    "<verify_proposal>\n<command>make check</command>\n</verify_proposal>"
)


def stub_verify(monkeypatch, results):
    queue = list(results)
    seen = []

    async def fake(command, workdir, **kwargs):
        seen.append(command)
        return queue.pop(0)

    monkeypatch.setattr(pump_module, "run_verify", fake)
    return seen


@pytest.mark.asyncio
async def test_goal_proposal_blocked_then_accept_by_index(workdir, monkeypatch):
    commands = stub_verify(monkeypatch, [VerifyResult(exit_code=0)])
    frontend, chunks, reporter = make_frontend(
        workdir,
        [
            [TextChunk(text=PROPOSAL)],  # proposer round
            [TextChunk(text="修复完成")],  # first working round after accept
        ],
    )

    async def script():
        yield Submit("/goal 修复所有测试")
        await wait_until(lambda: reporter.states[-1:] == ["blocked"])
        yield Submit("1")
        await wait_until(lambda: "目标达成" in output_of(chunks))
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    out = output_of(chunks)
    assert "1. make check" in out  # numbered proposal on screen
    assert "验收方式已确认：make check" in out
    assert commands == ["make check"]  # verify ran with the accepted command
    assert "✅ 目标达成" in out
    assert reporter.states == [
        "idle",
        "working",
        "idle",
        "blocked",
        "working",
        "idle",
    ]


@pytest.mark.asyncio
async def test_goal_proposal_skip_keeps_single_round(workdir):
    frontend, chunks, reporter = make_frontend(
        workdir, [[TextChunk(text=PROPOSAL)]]
    )

    async def script():
        yield Submit("/goal 做点事")
        await wait_until(lambda: reporter.states[-1:] == ["blocked"])
        yield Submit("/skip")
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    assert "已跳过自动验收" in output_of(chunks)
    assert reporter.states == ["idle", "working", "idle", "blocked", "idle"]


@pytest.mark.asyncio
async def test_goal_proposal_custom_command_is_the_edit_branch(
    workdir, monkeypatch
):
    commands = stub_verify(monkeypatch, [VerifyResult(exit_code=0)])
    frontend, chunks, reporter = make_frontend(
        workdir,
        [[TextChunk(text=PROPOSAL)], [TextChunk(text="好了")]],
    )

    async def script():
        yield Submit("/goal 验收")
        await wait_until(lambda: reporter.states[-1:] == ["blocked"])
        yield Submit("pytest -x")  # edited command, not a proposal index
        await wait_until(lambda: "目标达成" in output_of(chunks))
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    assert "验收方式已确认：pytest -x" in output_of(chunks)
    assert commands == ["pytest -x"]


@pytest.mark.asyncio
async def test_goal_proposal_interrupt_declines(workdir):
    frontend, chunks, reporter = make_frontend(
        workdir, [[TextChunk(text=PROPOSAL)]]
    )

    async def script():
        yield Submit("/goal 做点事")
        await wait_until(lambda: reporter.states[-1:] == ["blocked"])
        yield Interrupt(hard=False)  # Esc on the proposal prompt = decline
        yield Eof()

    rc = await frontend.run(script())

    assert rc == 0
    assert "已跳过自动验收" in output_of(chunks)
    assert reporter.states == ["idle", "working", "idle", "blocked", "idle"]


# -- /compact, /model, /new -----------------------------------------------------


@pytest.mark.asyncio
async def test_compact_runs_mini_turn(workdir):
    frontend, chunks, reporter = make_frontend(workdir, [])

    rc = await frontend.run(feed([Submit("/compact"), Eof()]))

    assert rc == 0
    assert "✗ 内部错误" not in output_of(chunks)
    assert reporter.states == ["idle", "working", "idle"]


@pytest.mark.asyncio
async def test_model_query_and_switch(workdir, tmp_path):
    # conftest points DEFAULT_CONFIG_PATH at this tmp_path; give the switch
    # an API key to validate against.
    config_path = config_module.DEFAULT_CONFIG_PATH
    config_path.write_text("[llm]\napi_key = 'fake-key'\n")
    frontend, chunks, _ = make_frontend(workdir, [])

    rc = await frontend.run(
        feed([Submit("/model"), Submit("/model foo-bar-1"), Eof()])
    )

    assert rc == 0
    out = output_of(chunks)
    assert "当前模型：" in out
    assert "已切换模型 foo-bar-1" in out


@pytest.mark.asyncio
async def test_new_session_reports_fresh_identity(workdir):
    frontend, chunks, reporter = make_frontend(workdir, [])

    rc = await frontend.run(feed([Submit("/new"), Eof()]))

    assert rc == 0
    sessions = [c for c in reporter.calls if c[0] == "session"]
    assert len(sessions) == 2  # startup + /new
    assert sessions[0][1] != sessions[1][1]  # a fresh session id
    assert "已开始新会话" in output_of(chunks)


# -- submission / verify-answer parsing (pure) -----------------------------------


def test_parse_submission_commands():
    assert parse_submission("hello") == Prompt("hello")
    assert parse_submission("/goal 修 bug") == SlashGoal("修 bug")
    assert parse_submission("/goal") == SlashGoal("")
    assert parse_submission("/MODEL glm") == SlashModel("glm")
    assert parse_submission("/new") == SlashNew()
    assert parse_submission("/quit") == SlashQuit()
    assert parse_submission("/exit") == SlashQuit()
    assert parse_submission("/bogus") == UnknownCommand("bogus")


def test_parse_submission_leading_space_forces_plain_text():
    assert parse_submission(" /new") == Prompt(" /new")


def test_parse_verify_answer():
    proposals = ["make check", "pytest -x"]
    assert parse_verify_answer("1", proposals) == AcceptVerify("make check")
    assert parse_verify_answer("2", proposals) == AcceptVerify("pytest -x")
    assert isinstance(parse_verify_answer("3", proposals), InvalidVerifyAnswer)
    assert parse_verify_answer("/skip", proposals) == SkipVerify()
    assert parse_verify_answer("", proposals) == SkipVerify()
    assert parse_verify_answer("pytest -v", proposals) == AcceptVerify("pytest -v")
    assert isinstance(
        parse_verify_answer("/goal clear", proposals), InvalidVerifyAnswer
    )


# -- terminal line parser (pure bytes) -------------------------------------------


def submits_of(actions):
    return [a for a in actions if isinstance(a, Submit)]


def test_parser_enter_submits_and_echoes():
    parser = TerminalLineParser()
    actions = parser.feed(b"hello\r")
    assert submits_of(actions) == [Submit("hello")]
    echo_text = "".join(a.text for a in actions if isinstance(a, Echo))
    assert echo_text == "hello\r\n"


def test_parser_utf8_input():
    parser = TerminalLineParser()
    assert submits_of(parser.feed("你好\r".encode())) == [Submit("你好")]


def test_parser_bracketed_paste_is_one_submission():
    parser = TerminalLineParser()
    before = parser.feed(b"\x1b[200~line1\nline2")
    assert submits_of(before) == []  # newline inside paste does not submit
    after = parser.feed(b"\x1b[201~\r")
    assert submits_of(after) == [Submit("line1\nline2")]


def test_parser_lone_esc_interrupts_after_flush():
    parser = TerminalLineParser()
    assert parser.feed(b"\x1b") == []
    assert parser.has_pending_escape
    assert parser.flush_escape() == [Interrupt(hard=False)]


def test_parser_arrow_sequence_is_dropped_not_esc():
    parser = TerminalLineParser()
    actions = parser.feed(b"\x1b[Aa\r")  # up-arrow, then "a", then Enter
    assert submits_of(actions) == [Submit("a")]


def test_parser_ctrl_c_and_ctrl_d():
    parser = TerminalLineParser()
    assert parser.feed(b"\x03") == [Interrupt(hard=True)]
    assert parser.feed(b"\x04") == [Eof()]
    # Ctrl+D with a non-empty line buffer is not EOF
    parser2 = TerminalLineParser()
    assert not [a for a in parser2.feed(b"a\x04") if isinstance(a, Eof)]


def test_parser_backspace_erases_multibyte_char():
    parser = TerminalLineParser()
    actions = parser.feed("ab你".encode() + b"\x7f\r")
    assert submits_of(actions) == [Submit("ab")]

# -- layering lock: the headless frontend stays textual-free ---------------------


def test_headless_imports_without_textual():
    import subprocess
    import sys

    code = (
        "import sys; import limbo.headless; "
        "bad = [m for m in sys.modules if m.startswith('textual')]; "
        "assert not bad, f'non-UI layer pulled in textual: {bad}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
