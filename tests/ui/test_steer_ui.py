"""Tests for mid-turn steer messaging UI (RFC LIM-20 + v2.1 cancel)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import Attachment, TextChunk, ToolCallEvent
from limbo.ui.app import LimboApp
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget, UserSubmitted
from limbo.ui.widgets.status_bar import StatusBar


class GatedClient:
    """LLM client replaying scripted events; each call may wait on a gate."""

    def __init__(self):
        self.calls = []
        self.script = []

    def add(self, events, gate=None):
        self.script.append((gate, events))

    async def chat(self, messages, tools, on_request=None):
        self.calls.append(list(messages))
        gate, events = self.script.pop(0)
        if gate is not None:
            await gate.wait()
        for event in events:
            yield event


def make_app(tmp_path: Path, client) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=client,
        session_dir=tmp_path / "sessions",
    )


def get_screen(pilot) -> MainScreen:
    return pilot.app.screen_stack[-1]


async def wait_until(pilot, predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError("timed out waiting for condition")


async def type_and_submit(pilot, input_widget: InputWidget, text: str) -> None:
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


def write_project_skill(workdir: Path, name: str, description: str, body: str) -> None:
    skill_dir = workdir / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


# -- queueing and delivery (Test Plan #12, #13, #19) ---------------------------


@pytest.mark.asyncio
async def test_submit_while_busy_queues_and_delivers(tmp_path):
    """Busy: input stays usable, Enter queues (no second worker), the card
    flips on SteerEvent, the status bar counts down to zero."""
    gate = asyncio.Event()
    client = GatedClient()
    client.add([ToolCallEvent(id="c1", name="read", arguments={"path": "main.py"})])
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])
    (tmp_path / "main.py").write_text("x = 1\n")
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)

        screen.run_worker(screen._handle_turn("read main.py"))
        await wait_until(pilot, lambda: len(client.calls) == 2)

        # Input stays usable mid-turn; no second LLM call on submit.
        assert screen._agent_busy
        assert not input_widget.disabled
        await type_and_submit(pilot, input_widget, "改用 uv")
        assert len(client.calls) == 2
        assert screen.agent.queued_count == 1
        assert "❯ 改用 uv（排队中）" in chat.transcript_text()
        assert "1 条排队中" in statusbar._context_text()

        gate.set()
        await wait_until(
            pilot, lambda: len(client.calls) == 3 and "done" in chat.transcript_text()
        )

        # Injected before the next request; card flipped; counter cleared.
        assert client.calls[2][-1].content == "改用 uv"
        transcript = chat.transcript_text()
        assert "❯ 改用 uv" in transcript
        assert "（排队中）" not in transcript
        assert "排队中" not in statusbar._context_text()
        assert not screen._agent_busy


# -- busy command guards (Test Plan #14, #15) ------------------------------------


@pytest.mark.asyncio
async def test_busy_rejects_history_commands(tmp_path):
    client = GatedClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        chat.add_info("标记消息")
        screen.driver._running = True  # simulate an in-flight turn (driver owns single-flight)
        agent_before = screen.agent

        screen._handle_command("/new")
        screen._handle_command("/sessions")
        screen._handle_command("/export")
        screen._handle_command("/compact")
        await pilot.pause()

        transcript = chat.transcript_text()
        assert transcript.count("当前任务进行中") == 4
        # /compact keeps its pre-existing wording.
        assert "请等待完成后再压缩" in transcript
        # Agent not replaced, chat not cleared.
        assert screen.agent is agent_before
        assert "标记消息" in transcript
        assert client.calls == []

        # Read-only commands stay available while busy.
        screen._handle_command("/help")
        await pilot.pause()
        assert "可用命令" in chat.transcript_text()
        screen._handle_command("/2048")
        await pilot.pause()
        assert pilot.app.screen_stack[-1] is not screen
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_busy_rejection_via_enter_submission(tmp_path):
    """The slash-menu Enter path funnels into the same central guard."""
    client = GatedClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)
        screen.driver._running = True  # simulate an in-flight turn (driver owns single-flight)
        agent_before = screen.agent

        input_widget.text = "/new"
        await pilot.press("enter")
        await pilot.pause()

        assert "当前任务进行中" in chat.transcript_text()
        assert screen.agent is agent_before


# -- skill queueing (Test Plan #16) ----------------------------------------------


@pytest.mark.asyncio
async def test_skill_invocation_while_busy_enqueues_full_prompt(tmp_path):
    write_project_skill(tmp_path, "tdd", "Test-driven development.", "# TDD rules\nRed first.")
    client = GatedClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        screen.driver._running = True  # simulate an in-flight turn (driver owns single-flight)

        screen._handle_command("/tdd implement login")
        await pilot.pause()

        # The assembled prompt (not the raw input) is queued; no worker, no call.
        assert screen.agent.queued_count == 1
        item = next(iter(screen.agent._steer_queue))
        assert "# TDD rules" in item.text
        assert "Red first." in item.text
        assert "implement login" in item.text
        assert "❯ /tdd implement login（排队中）" in chat.transcript_text()
        assert client.calls == []


# -- error-exit auto follow-up (Test Plan #17) -----------------------------------


@pytest.mark.asyncio
async def test_error_exit_auto_followup_turn(tmp_path):
    """A queued message survives an error exit: the turn worker's finally
    drains it into an automatic follow-up turn (busy never drops in between)."""
    gate = asyncio.Event()

    class ErrorOnceClient:
        def __init__(self):
            self.calls = []

        async def chat(self, messages, tools, on_request=None):
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                await gate.wait()
                raise RuntimeError("boom")
            yield TextChunk(text="recovered")

    client = ErrorOnceClient()
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("hi"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        await type_and_submit(pilot, input_widget, "排队消息")
        assert screen.agent.queued_count == 1

        gate.set()
        await wait_until(
            pilot,
            lambda: len(client.calls) == 2 and "recovered" in chat.transcript_text(),
        )

        transcript = chat.transcript_text()
        assert "boom" in transcript
        # The follow-up turn carried the queued message as its input.
        assert client.calls[1][-1].content == "排队消息"
        assert "（排队中）" not in transcript
        assert not screen.agent.has_pending_steer()
        assert not screen._agent_busy


# -- attachments (Test Plan #18) ---------------------------------------------------


@pytest.mark.asyncio
async def test_queued_message_with_attachment(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("附件内容")
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)

        attachment = Attachment(kind="file", name="note.txt", path=str(note))
        screen.on_user_submitted(UserSubmitted("看附件", [attachment]))
        await pilot.pause()

        assert screen.agent.queued_count == 1
        assert next(iter(screen.agent._steer_queue)).attachments == [attachment]
        assert "📎 文件: note.txt" in chat.transcript_text()

        gate.set()
        await wait_until(pilot, lambda: "done" in chat.transcript_text())
        steer_msg = client.calls[1][-1]
        assert "附件内容" in steer_msg.content


# -- cancel: Esc (Test Plan #25, #26, #30) ----------------------------------------


@pytest.mark.asyncio
async def test_esc_cancels_latest_queued(tmp_path):
    """RFC LIM-53: while a turn runs, ESC interrupts the turn; cancelling
    the newest queued steer is what ESC does once no turn is running."""
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="done")], gate=gate)
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)
        statusbar = screen.query_one("#statusbar", StatusBar)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        await type_and_submit(pilot, input_widget, "第一条")
        await type_and_submit(pilot, input_widget, "第二条")
        assert screen.agent.queued_count == 2

        # ESC while the turn runs interrupts the turn; the queue survives.
        await pilot.press("escape")
        await wait_until(pilot, lambda: not screen.driver.running)
        assert screen.agent.queued_count == 2
        assert "⏹ 已打断" in chat.transcript_text()

        # Turn over: Esc cancels the newest queued message first (LIFO).
        await pilot.press("escape")
        await pilot.pause()
        assert screen.agent.queued_count == 1
        transcript = chat.transcript_text()
        assert "❯ 第二条（已取消）" in transcript
        assert "❯ 第一条（排队中）" in transcript
        assert "1 条排队中" in statusbar._context_text()

        await pilot.press("escape")
        await pilot.pause()
        assert screen.agent.queued_count == 0
        assert "排队中" not in statusbar._context_text()

        # Esc on an empty queue is a no-op.
        await pilot.press("escape")
        await pilot.pause()

        # Nothing was ever injected (the interrupted turn drained nothing).
        user_texts = [m.content for m in screen.agent.messages if m.role == "user"]
        assert user_texts == ["go"]


@pytest.mark.asyncio
async def test_esc_closes_slash_menu_before_cancelling(tmp_path):
    """The slash menu keeps Esc priority: menu open → close menu only."""
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="done")], gate=gate)
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        await type_and_submit(pilot, input_widget, "排队")
        assert screen.agent.queued_count == 1

        await pilot.press("/")
        await pilot.pause()
        assert screen.slash_menu_open

        await pilot.press("escape")
        await pilot.pause()
        assert not screen.slash_menu_open
        # The queue is untouched while the menu handled Esc.
        assert screen.agent.queued_count == 1
        assert "❯ 排队（排队中）" in chat.transcript_text()

        # Menu closed now: the turn is still running, so the next Esc
        # interrupts the turn (RFC LIM-53) instead of touching the queue.
        await pilot.press("escape")
        await wait_until(pilot, lambda: not screen.driver.running)
        assert screen.agent.queued_count == 1
        assert "❯ 排队（排队中）" in chat.transcript_text()

        # Turn over: Esc reaches the queue.
        await pilot.press("escape")
        await pilot.pause()
        assert screen.agent.queued_count == 0
        assert "❯ 排队（已取消）" in chat.transcript_text()


# -- cancel: card ✕ (Test Plan #27, #28) ------------------------------------------


@pytest.mark.asyncio
async def test_card_cancel_button_middle_item(tmp_path):
    """Cancelling a middle card does not misalign the remaining deliveries
    (id-based pairing, not FIFO order)."""
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        for text in ("一", "二", "三"):
            await type_and_submit(pilot, input_widget, text)
        assert screen.agent.queued_count == 3

        middle = list(chat.queued_cards.values())[1]
        await pilot.click(middle.query_one(".queued-cancel"))
        await pilot.pause()
        assert screen.agent.queued_count == 2
        assert "❯ 二（已取消）" in chat.transcript_text()

        gate.set()
        await wait_until(pilot, lambda: "done" in chat.transcript_text())

        transcript = chat.transcript_text()
        assert "❯ 一（排队中）" not in transcript and "❯ 一" in transcript
        assert "❯ 三（排队中）" not in transcript and "❯ 三" in transcript
        user_texts = [m.content for m in screen.agent.messages if m.role == "user"]
        assert user_texts == ["go", "一", "三"]


@pytest.mark.asyncio
async def test_cancel_after_delivery_is_refused(tmp_path):
    """Past the injection boundary a message cannot be retracted."""
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        await type_and_submit(pilot, input_widget, "会注入")
        item_id = next(iter(chat.queued_cards))

        gate.set()
        await wait_until(pilot, lambda: "done" in chat.transcript_text())
        assert not chat.queued_cards  # delivered

        screen.cancel_queued(item_id)
        await pilot.pause()
        assert "该消息已注入，无法撤回" in chat.transcript_text()
        user_texts = [m.content for m in screen.agent.messages if m.role == "user"]
        assert user_texts == ["go", "会注入"]


# -- cancel: skill prompt (Test Plan #29) ------------------------------------------


@pytest.mark.asyncio
async def test_skill_queued_cancel(tmp_path):
    write_project_skill(tmp_path, "tdd", "Test-driven development.", "# TDD rules")
    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])
    app = make_app(tmp_path, client)
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)
        screen._handle_command("/tdd refactor")
        await pilot.pause()
        assert screen.agent.queued_count == 1

        item_id = next(iter(chat.queued_cards))
        screen.cancel_queued(item_id)
        await pilot.pause()
        assert screen.agent.queued_count == 0
        assert "❯ /tdd refactor（已取消）" in chat.transcript_text()

        gate.set()
        await wait_until(pilot, lambda: "mid" in chat.transcript_text())
        # The turn ended normally — no follow-up call, and the model never
        # saw the skill prompt.
        assert len(client.calls) == 1
        for call in client.calls:
            assert not any(
                m.role == "user" and "TDD rules" in (m.content or "") for m in call
            )


# -- interaction with LIM-19 queued steer --------------------------------------


@pytest.mark.asyncio
async def test_queued_steer_message_delivered_verbatim(tmp_path):
    """A busy submission is queued verbatim and delivered as the next turn's
    user message once the in-flight turn ends."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("external")
    workdir = tmp_path / "workdir"
    workdir.mkdir()

    gate = asyncio.Event()
    client = GatedClient()
    client.add([TextChunk(text="mid")], gate=gate)
    client.add([TextChunk(text="done")])

    cfg = Config()
    cfg.llm.api_key = "test"
    app = LimboApp(
        workdir=workdir,
        config=cfg,
        llm_client=client,
        session_dir=tmp_path / "sessions",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = get_screen(pilot)
        chat = screen.query_one("#chat", ChatWidget)
        input_widget = screen.query_one("#input", InputWidget)

        screen.run_worker(screen._handle_turn("go"))
        await wait_until(pilot, lambda: len(client.calls) == 1)

        await type_and_submit(pilot, input_widget, f"顺便看看 {outside}")
        assert screen.agent.queued_count == 1

        gate.set()
        await wait_until(pilot, lambda: len(client.calls) == 2 and "done" in chat.transcript_text())
        assert client.calls[1][-1].content == f"顺便看看 {outside}"
