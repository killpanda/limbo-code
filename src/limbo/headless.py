"""Headless CLI frontend (RFC LIM-60, design/rfc-headless-cli.md).

A plain terminal REPL frontend — the second Limbo frontend alongside the
Textual TUI — built so the Herdr multiplexer can drive it with its native
control surface:

- ``herdr agent prompt`` writes "text + Enter" into the pane (honoring the
  pane's live bracketed-paste mode) → this REPL reads line-oriented stdin,
  enables bracketed paste (\\e[?2004h), and echoes input itself (cbreak
  mode disables kernel echo, and Herdr reads the screen to observe).
- ``herdr agent send-keys esc|ctrl+c`` → Esc interrupts the current turn;
  Ctrl+C interrupts while busy and quits while idle.
- ``herdr agent read`` sees only the primary screen (Textual's alternate
  screen is unreadable to it) → everything renders as plain stdout text.
- ``herdr agent wait --until blocked`` syncs on the lifecycle state this
  frontend reports through the existing integrations layer.

Non-UI layer: this module must never import textual. All orchestration
(single-flight, steer drain, goal loop) stays in ``TurnPump``/``Agent`` —
the frontend only translates stdin into pump calls and pump events into
stdout text.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import unicodedata
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from limbo.agent import (
    Agent,
    CompactionEvent,
    ErrorEvent,
    InterruptEvent,
    SteerEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallRequest,
    ToolResultEvent,
    UsageUpdate,
)
from limbo.config import Config
from limbo.goal import (
    STATUS_ACTIVE,
    GoalClear,
    GoalQuery,
    GoalSet,
    build_initial_prompt,
    build_proposer_prompt,
    parse_goal_args,
    parse_verify_proposal,
)
from limbo.integrations import create_reporters, install_exit_hooks
from limbo.integrations.base import AgentState, IntegrationReporter
from limbo.llm.client import LLMClient
from limbo.llm.factory import create_llm_client
from limbo.model_switch import (
    prepare_model_switch,
    reload_llm_config,
    swap_llm_client,
)
from limbo.pump import (
    GoalExhausted,
    GoalPassed,
    GoalResumed,
    GoalVerifyResultEvent,
    GoalVerifyStarted,
    PumpEvent,
    TurnPump,
)

# -- input events (stdin → frontend) ------------------------------------------


@dataclass(frozen=True)
class Submit:
    """One submitted input line (a prompt or a /command)."""

    text: str


@dataclass(frozen=True)
class Interrupt:
    """Esc (``hard=False``) or Ctrl+C (``hard=True``).

    Both interrupt a running turn; only Ctrl+C quits from idle.
    """

    hard: bool = False


@dataclass(frozen=True)
class Eof:
    """stdin closed: finish the in-flight turn, then exit."""


InputEvent = Submit | Interrupt | Eof


@dataclass(frozen=True)
class _TurnDone:
    """Internal queue event: the turn/compact worker task finished."""

    error: BaseException | None = None


# -- submission parsing (plain text vs /commands) ------------------------------


@dataclass(frozen=True)
class Prompt:
    text: str


@dataclass(frozen=True)
class SlashGoal:
    arg: str


@dataclass(frozen=True)
class SlashCompact:
    pass


@dataclass(frozen=True)
class SlashModel:
    arg: str


@dataclass(frozen=True)
class SlashNew:
    pass


@dataclass(frozen=True)
class SlashHelp:
    pass


@dataclass(frozen=True)
class SlashQuit:
    pass


@dataclass(frozen=True)
class UnknownCommand:
    name: str


Submission = (
    Prompt
    | SlashGoal
    | SlashCompact
    | SlashModel
    | SlashNew
    | SlashHelp
    | SlashQuit
    | UnknownCommand
)


def parse_submission(text: str) -> Submission:
    """Classify one submitted line: a prompt or a slash command.

    A leading space forces plain text (same convention as the TUI's
    input widget), so "/path-like" messages can still be sent.
    """
    if not text.startswith("/"):
        return Prompt(text)
    stripped = text.strip()
    name, _, arg = stripped[1:].partition(" ")
    name = name.lower()
    arg = arg.strip()
    if name == "goal":
        return SlashGoal(arg)
    if name == "compact":
        return SlashCompact()
    if name == "model":
        return SlashModel(arg)
    if name == "new":
        return SlashNew()
    if name == "help":
        return SlashHelp()
    if name in ("quit", "exit"):
        return SlashQuit()
    return UnknownCommand(name)


# -- verify-proposal answer parsing (the blocked state) ------------------------


@dataclass(frozen=True)
class AcceptVerify:
    command: str


@dataclass(frozen=True)
class SkipVerify:
    pass


@dataclass(frozen=True)
class InvalidVerifyAnswer:
    reason: str


VerifyAnswer = AcceptVerify | SkipVerify | InvalidVerifyAnswer


def parse_verify_answer(text: str, proposals: list[str]) -> VerifyAnswer:
    """Interpret the submission that arrives while blocked on a proposal.

    A 1-based index picks a proposal; ``/skip`` (or an empty line) declines
    auto-verify; any other plain line is the (edited) verify command —
    that is the headless equivalent of the TUI picker's edit branch.
    """
    stripped = text.strip()
    if not stripped or stripped == "/skip":
        return SkipVerify()
    if stripped.startswith("/"):
        return InvalidVerifyAnswer(
            "正在等待验收确认：回复序号、验收命令，或 /skip 跳过"
        )
    if stripped.isdigit():
        index = int(stripped)
        if 1 <= index <= len(proposals):
            return AcceptVerify(proposals[index - 1])
        return InvalidVerifyAnswer(
            f"序号超出范围（1–{len(proposals)}）；或直接输入验收命令，/skip 跳过"
        )
    return AcceptVerify(stripped)


# -- terminal line parser (TTY byte stream → input events) ---------------------


@dataclass(frozen=True)
class Echo:
    """Bytes to write back to the terminal (cbreak disables kernel echo)."""

    text: str


ParsedInput = Submit | Interrupt | Eof | Echo

_ESC = 0x1B
_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"
_KNOWN_SEQUENCES = (_PASTE_START, _PASTE_END)


def _char_width(cell_text: str) -> int:
    """Terminal cell width of one char (for backspace erasure)."""
    return 2 if unicodedata.east_asian_width(cell_text) in ("W", "F") else 1


class TerminalLineParser:
    """Incremental byte parser for the cbreak-mode TTY reader.

    Handles: printable UTF-8 input (echoed back), Enter → ``Submit``,
    Backspace erasure (width-aware for CJK), bracketed paste blocks
    (embedded newlines stay in the buffer; the Enter herdr sends after the
    paste-end marker submits), Ctrl+C → hard Interrupt, Ctrl+D on an empty
    line → ``Eof``. A trailing lone ESC is held in ``_escape`` until more
    bytes disambiguate it from a sequence, or the reader calls
    ``flush_escape()`` after a short timeout (that is the Esc key).
    Unknown escape sequences (arrows, alt- combos) are dropped.
    """

    def __init__(self) -> None:
        self._line = bytearray()
        self._escape: bytearray | None = None
        self._in_paste = False

    @property
    def has_pending_escape(self) -> bool:
        return self._escape is not None

    def flush_escape(self) -> list[ParsedInput]:
        """Resolve a held escape after the disambiguation timeout."""
        escape = self._escape
        self._escape = None
        if escape == bytearray([_ESC]):
            return [Interrupt(hard=False)]
        return []

    def feed(self, data: bytes) -> list[ParsedInput]:
        actions: list[ParsedInput] = []
        echo = bytearray()

        def flush_echo() -> None:
            if echo:
                actions.append(Echo(echo.decode("utf-8", errors="replace")))
                echo.clear()

        for byte in data:
            if self._escape is not None:
                self._escape.append(byte)
                seq = bytes(self._escape)
                if seq == _PASTE_START:
                    self._in_paste = True
                    self._escape = None
                elif seq == _PASTE_END:
                    self._in_paste = False
                    self._escape = None
                elif any(s.startswith(seq) for s in _KNOWN_SEQUENCES):
                    pass  # need more bytes to decide
                elif len(seq) >= 2 and seq[1:2] == b"[":
                    # Unknown CSI: consume until a final byte (0x40–0x7E).
                    if 0x40 <= byte <= 0x7E:
                        self._escape = None
                else:
                    # ESC + non-CSI byte (alt- combo): drop both.
                    self._escape = None
                continue
            if byte == _ESC:
                self._escape = bytearray([byte])
                continue
            if self._in_paste:
                self._line.append(byte)
                echo.extend(b"\r\n" if byte == 0x0A else bytes([byte]))
                continue
            if byte == 0x03:  # Ctrl+C (ISIG is off in cbreak mode)
                flush_echo()
                actions.append(Interrupt(hard=True))
            elif byte == 0x04:  # Ctrl+D: EOF on an empty line only
                if not self._line:
                    flush_echo()
                    actions.append(Eof())
            elif byte in (0x0D, 0x0A):  # Enter submits
                text = bytes(self._line).decode("utf-8", errors="replace")
                self._line.clear()
                flush_echo()
                actions.append(Echo("\r\n"))
                actions.append(Submit(text))
            elif byte in (0x7F, 0x08):  # Backspace
                if self._line:
                    popped = bytearray()
                    while self._line and self._line[-1] & 0xC0 == 0x80:
                        popped.insert(0, self._line.pop())
                    popped.insert(0, self._line.pop())
                    cell = bytes(popped).decode("utf-8", errors="ignore")
                    echo.extend(b"\b \b" * _char_width(cell))
            elif byte >= 0x20 or byte == 0x09:  # printable / tab
                self._line.append(byte)
                echo.append(byte)
            # remaining control bytes are dropped
        flush_echo()
        return actions


# -- stdin readers --------------------------------------------------------------


async def read_pipe(stream) -> AsyncIterator[InputEvent]:
    """Line reader for non-TTY stdin (scripts, tests): each line submits."""
    while True:
        line = await asyncio.to_thread(stream.readline)
        if line == "":
            yield Eof()
            return
        yield Submit(line.rstrip("\r\n"))


async def read_tty(fd: int, out: Callable[[str], None]) -> AsyncIterator[InputEvent]:
    """Byte reader for TTY stdin: cbreak mode + bracketed paste + echo."""
    parser = TerminalLineParser()
    while True:
        if parser.has_pending_escape:
            try:
                data = await asyncio.wait_for(
                    asyncio.to_thread(os.read, fd, 4096), timeout=0.08
                )
            except TimeoutError:
                data = None
        else:
            data = await asyncio.to_thread(os.read, fd, 4096)
        actions = parser.flush_escape() if data is None else parser.feed(data)
        if data == b"":
            yield Eof()
            return
        for action in actions:
            if isinstance(action, Echo):
                out(action.text)
            else:
                yield action


@contextlib.contextmanager
def _terminal_mode(fd: int) -> Iterator[None]:
    """cbreak + no echo + no ISIG, and enable bracketed paste (\\e[?2004h).

    ISIG off makes Ctrl+C arrive as byte 0x03 the parser can route to
    interrupt/quit instead of killing the process. Herdr's ``agent prompt``
    detects the 2004 mode and wraps multiline text in paste markers so
    embedded newlines don't submit early.
    """
    import termios  # POSIX-only; callers gate on os.name / isatty

    saved = termios.tcgetattr(fd)
    raw = termios.tcgetattr(fd)
    raw[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)  # lflag
    raw[6][termios.VMIN] = 1
    raw[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSADRAIN, raw)
    sys.stdout.write("\x1b[?2004h")
    sys.stdout.flush()
    try:
        yield
    finally:
        sys.stdout.write("\x1b[?2004l")
        sys.stdout.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# -- the frontend -----------------------------------------------------------------

HELP_TEXT = """可用命令：
  /goal <目标>   设定目标并进入闭环（首轮提议验收命令）
  /goal clear    退出 goal 模式
  /compact       立即压缩对话历史
  /model [id]    查看/切换模型
  /new           开始新会话
  /quit          退出
忙时输入的普通文本会作为插话排队；Esc / Ctrl+C 中断当前任务。"""


def _first_line(text: str, limit: int = 60) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class HeadlessFrontend:
    """Drives one session's TurnPump from a stream of InputEvents.

    Everything is serialized through one asyncio queue: the stdin reader
    (or a test's async iterator) feeds ``Submit``/``Interrupt``/``Eof``,
    and the turn worker's done-callback feeds ``_TurnDone`` — so goal
    post-turn logic, busy checks, and EOF handling never race.
    """

    def __init__(
        self,
        *,
        config: Config,
        agent: Agent,
        pump: TurnPump,
        llm_client: LLMClient,
        reporters: IntegrationReporter,
        out: Callable[[str], None],
        agent_factory: Callable[[LLMClient], Agent],
        pump_factory: Callable[[Agent], TurnPump],
        echo_input: bool = False,
        max_tool_lines: int = 60,
    ) -> None:
        self._config = config
        self._agent = agent
        self._pump = pump
        self._llm_client = llm_client
        self._reporters = reporters
        self._out = out
        self._agent_factory = agent_factory
        self._pump_factory = pump_factory
        self._echo_input = echo_input
        self._max_tool_lines = max_tool_lines

        self._queue: asyncio.Queue[InputEvent | _TurnDone] = asyncio.Queue()
        self._turn_task: asyncio.Task[None] | None = None
        # Goal proposal state: _proposal_pending = a proposer round is in
        # flight or just ended (parse its output); _awaiting_verify = the
        # proposals are on screen and the next submission is the answer.
        self._proposal_pending = False
        self._awaiting_verify = False
        self._pending_proposals: list[str] = []
        self._eof = False
        self._quit = False
        self._last_tokens = 0

    # -- main loop ------------------------------------------------------------

    async def run(self, inputs: AsyncIterator[InputEvent]) -> int:
        forwarder = asyncio.create_task(self._forward(inputs))
        self._report_session()
        self._report("idle")
        self._line(
            f"limbo headless · model {self._config.llm.model} · "
            f"workdir {self._agent.workdir}"
        )
        self._line("输入消息回车发送；/help 查看命令；Esc/Ctrl+C 中断当前任务")
        try:
            while not self._quit:
                event = await self._queue.get()
                if isinstance(event, _TurnDone):
                    self._on_turn_done(event)
                elif isinstance(event, Submit):
                    await self._on_submit(event.text)
                elif isinstance(event, Interrupt):
                    self._on_interrupt(event)
                elif isinstance(event, Eof):
                    self._on_eof()
        finally:
            forwarder.cancel()
            if self._turn_task is not None:
                self._turn_task.cancel()
            self._reporters.release()
            self._agent.close()
        return 0

    async def _forward(self, inputs: AsyncIterator[InputEvent]) -> None:
        try:
            async for event in inputs:
                self._queue.put_nowait(event)
        finally:
            self._queue.put_nowait(Eof())

    # -- input dispatch -------------------------------------------------------

    async def _on_submit(self, text: str) -> None:
        if self._echo_input:
            self._line(f"❯ {text}")
        if self._awaiting_verify:
            self._answer_verify(text)
            return
        command = parse_submission(text)
        busy = self._turn_task is not None
        if busy:
            # Same discipline as the TUI (RFC LIM-20): plain text steers,
            # history-shaping commands are refused while a turn owns the
            # agent; /help stays available.
            if isinstance(command, Prompt):
                self._agent.steer(text)
                self._line(f"➤ 已排队插话：{_first_line(text)}")
            elif isinstance(command, SlashHelp):
                self._line(HELP_TEXT)
            else:
                self._line(
                    "当前任务进行中，该命令暂不可用"
                    "（Esc/Ctrl+C 中断，或等待完成）"
                )
            return
        if isinstance(command, Prompt):
            self._start(self._pump.run(text))
        elif isinstance(command, SlashGoal):
            self._cmd_goal(command.arg)
        elif isinstance(command, SlashCompact):
            self._start(self._pump.compact())
        elif isinstance(command, SlashModel):
            await self._cmd_model(command.arg)
        elif isinstance(command, SlashNew):
            self._cmd_new()
        elif isinstance(command, SlashHelp):
            self._line(HELP_TEXT)
        elif isinstance(command, SlashQuit):
            self._quit = True
        elif isinstance(command, UnknownCommand):
            self._line(f"未知命令 /{command.name}（/help 查看可用命令）")

    def _on_interrupt(self, event: Interrupt) -> None:
        if self._awaiting_verify:
            # Esc/Ctrl+C on the proposal prompt = decline (picker cancel).
            self._awaiting_verify = False
            self._line("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            self._report("idle")
            return
        if self._turn_task is not None:
            self._agent.interrupt()
            self._pump.cancel_verify()
            return
        if event.hard:  # Ctrl+C at idle quits; Esc at idle is a no-op.
            self._quit = True

    def _on_eof(self) -> None:
        self._eof = True
        if self._awaiting_verify:
            self._awaiting_verify = False
            self._line("输入已关闭，跳过验收确认（保持单轮模式）")
            self._report("idle")
            self._quit = True
            return
        if self._turn_task is not None:
            self._line("输入已关闭，等待当前任务完成…")
            return
        self._quit = True

    # -- turn worker ----------------------------------------------------------

    def _start(self, events: AsyncIterator[PumpEvent]) -> None:
        # pump.run()/compact() set the busy flag eagerly at call time, so
        # single-flight holds even before the task first runs.
        self._turn_task = asyncio.create_task(self._pump_events(events))
        self._turn_task.add_done_callback(self._turn_finished)

    async def _pump_events(self, events: AsyncIterator[PumpEvent]) -> None:
        self._report("working")
        try:
            async for event in events:
                self._render(event)
        except Exception as e:  # noqa: BLE001 - mirror the TUI: surface it
            self._line(f"✗ 内部错误：{e}")
        finally:
            self._line(f"── limbo idle · 累计 tokens {self._last_tokens} ──")
            self._report("idle")

    def _turn_finished(self, task: asyncio.Task[None]) -> None:
        error = None if task.cancelled() else task.exception()
        self._queue.put_nowait(_TurnDone(error))

    def _on_turn_done(self, done: _TurnDone) -> None:
        self._turn_task = None
        if done.error is not None:
            self._line(f"✗ 内部错误：{done.error}")
        self._maybe_confirm_verify_proposal()
        if self._eof:
            if self._awaiting_verify:
                self._awaiting_verify = False
                self._line("输入已关闭，跳过验收确认（保持单轮模式）")
                self._report("idle")
            self._quit = True

    # -- /goal & the verify-proposal blocked state ----------------------------

    def _cmd_goal(self, arg: str) -> None:
        command = parse_goal_args(arg)
        if isinstance(command, GoalQuery):
            state = self._pump.status()
            if state is None or state.status == "cleared":
                self._line("当前没有 goal")
            else:
                self._line(
                    f"目标：{state.text}\n"
                    f"验收命令：{state.verify_command or '（未设置）'}\n"
                    f"进度：{state.rounds_completed}/{state.max_rounds} 轮 · "
                    f"状态 {state.status}"
                )
            return
        if isinstance(command, GoalClear):
            if self._pump.status() is None:
                self._line("当前没有 goal")
                return
            self._pump.clear()
            self._proposal_pending = False
            self._awaiting_verify = False
            self._line("已退出 goal 模式")
            return
        # GoalSet: the first round is a proposal round; the confirmation
        # happens when that turn ends (see _maybe_confirm_verify_proposal).
        assert isinstance(command, GoalSet)
        state = self._pump.set_goal(command.text)
        self._proposal_pending = True
        self._start(self._pump.run(build_proposer_prompt(state)))

    def _maybe_confirm_verify_proposal(self) -> None:
        """After a proposal round, present the model's verify command(s).

        Mirrors the TUI's picker: parse the last assistant message for
        <verify_proposal> at a quiet turn boundary, then block on the next
        submission as the answer. ``blocked`` is what
        ``herdr agent wait --until blocked`` syncs on.
        """
        if not self._proposal_pending:
            return
        self._proposal_pending = False
        state = self._pump.status()
        if state is None or state.status != STATUS_ACTIVE or state.verify_command:
            return
        text = ""
        for message in reversed(self._agent.messages):
            if message.role == "assistant" and message.content:
                text = message.content
                break
        parsed = parse_verify_proposal(text)
        if parsed is not None and len(parsed) == 0:
            # Explicit <none/>: no objective gate exists for this goal.
            self._line(
                "模型认为该目标没有客观可判定的验收方式，保持单轮模式；"
                "/goal clear 退出"
            )
            return
        self._pending_proposals = parsed or []
        if self._pending_proposals:
            self._line("模型提议的验收命令：")
            for index, proposal in enumerate(self._pending_proposals, 1):
                self._line(f"  {index}. {proposal}")
        else:
            self._line("未能解析模型的验收提议。")
        self._line("回复序号采纳；或直接输入一行作为验收命令；/skip 跳过自动验收")
        self._awaiting_verify = True
        self._report("blocked", message="等待确认 /goal 验收命令")

    def _answer_verify(self, text: str) -> None:
        answer = parse_verify_answer(text, self._pending_proposals)
        if isinstance(answer, InvalidVerifyAnswer):
            self._line(answer.reason)
            return
        self._awaiting_verify = False
        self._pending_proposals = []
        state = self._pump.status()
        if state is None or state.status != STATUS_ACTIVE:
            self._report("idle")
            return
        if isinstance(answer, SkipVerify):
            self._line("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            self._report("idle")
            return
        new_state = self._pump.set_verify(answer.command)
        if new_state is None:
            self._report("idle")
            return
        self._line(f"验收方式已确认：{answer.command}")
        self._start(self._pump.run(build_initial_prompt(new_state)))

    # -- other commands -------------------------------------------------------

    async def _cmd_model(self, arg: str) -> None:
        reload_llm_config(self._config)
        if not arg:
            self._line(f"当前模型：{self._config.llm.model}")
            return
        verdict = prepare_model_switch(arg, self._config)
        for notice in verdict.notices:
            self._line(notice)
        if not verdict.switched:
            return
        self._llm_client, notices = await swap_llm_client(
            self._config, self._llm_client, self._agent
        )
        for notice in notices:
            self._line(notice)

    def _cmd_new(self) -> None:
        self._agent.close()  # release the old session's trace file handle
        self._agent = self._agent_factory(self._llm_client)
        self._pump = self._pump_factory(self._agent)
        self._report_session()
        self._line("已开始新会话")

    # -- rendering ------------------------------------------------------------

    def _line(self, text: str = "") -> None:
        self._out(text + "\n")

    def _render(self, event: PumpEvent) -> None:
        if isinstance(event, TextDelta):
            self._out(event.text)
        elif isinstance(event, ThinkingDelta):
            self._out(event.text)
        elif isinstance(event, ToolCallRequest):
            # run_code's `code` argument is the whole program (huge, useless
            # as a summary): show the model's own description instead — that
            # is what a reader (human or herdr) wants on one line.
            if event.name == "run_code":
                description = event.arguments.get("description")
                if isinstance(description, str) and description:
                    self._line(f"\n⚙ run_code · {_first_line(description)}")
                    return
            args = json.dumps(event.arguments, ensure_ascii=False)
            if len(args) > 100:
                args = args[:97] + "..."
            self._line(f"\n⚙ {event.name}({args})")
        elif isinstance(event, ToolResultEvent):
            result = event.result
            if result.success:
                self._line(f"  ✓ {event.name}")
                output = result.output or ""
                lines = output.splitlines()
                if len(lines) > self._max_tool_lines:
                    hidden = len(lines) - self._max_tool_lines
                    output = "\n".join(lines[: self._max_tool_lines])
                    self._line(_indent(output))
                    self._line(f"  …（省略 {hidden} 行）")
                elif output:
                    self._line(_indent(output))
            else:
                self._line(f"  ✗ {event.name}: {result.error or 'failed'}")
        elif isinstance(event, ErrorEvent):
            self._line(f"✗ {event.message}")
        elif isinstance(event, CompactionEvent):
            if event.compacted:
                self._line(
                    f"已压缩上下文：约 {event.before_tokens} → "
                    f"{event.after_estimate} tokens"
                    + ("（手动）" if event.trigger == "manual" else "（自动）")
                )
            elif event.reason:
                self._line(event.reason)
            if event.warning:
                self._line(event.warning)
        elif isinstance(event, UsageUpdate):
            self._last_tokens = event.total_tokens
        elif isinstance(event, SteerEvent):
            self._line(f"➤ 插话已注入：{_first_line(event.text)}")
        elif isinstance(event, InterruptEvent):
            self._line("⏹ 已打断")
        elif isinstance(event, GoalVerifyStarted):
            self._line(
                f"🔍 第 {event.round}/{event.max_rounds} 轮验收："
                f"`{event.command}`"
            )
        elif isinstance(event, GoalVerifyResultEvent):
            vresult = event.result
            if vresult.cancelled:
                self._line("验收已取消，闭环暂停（goal 保持 active）")
            elif vresult.timed_out:
                self._line("✗ 验收命令执行超时，将带入下一轮处理")
            elif vresult.exit_code == 0:
                self._line("✅ 验收命令退出码 0")
            else:
                self._line(
                    f"❌ 第 {event.round} 轮验收未通过"
                    f"（退出码 {vresult.exit_code}），失败输出将原样注入下一轮"
                )
        elif isinstance(event, GoalPassed):
            self._line(f"✅ 目标达成（共 {event.rounds} 轮验收）")
        elif isinstance(event, GoalExhausted):
            self._line(
                f"⚠️ 已达最大轮次（{event.rounds} 轮）："
                "发送任意消息带新预算继续，/goal clear 退出"
            )
        elif isinstance(event, GoalResumed):
            self._line(f"🎯 恢复 goal 闭环，预算已重置（{event.max_rounds} 轮）")

    # -- integration reporting ------------------------------------------------

    def _report(self, state: AgentState, *, message: str | None = None) -> None:
        self._reporters.report(state, message=message)

    def _report_session(self) -> None:
        self._reporters.report_session(
            self._agent.session_id, str(self._agent.session_path)
        )


# -- entry point ------------------------------------------------------------------


def run_headless(
    *,
    workdir: Path,
    config: Config,
    session_dir: Path | None,
    resume: Path | None,
) -> int:
    """Synchronous entry for ``limbo --headless`` (called from app.main)."""
    try:
        return asyncio.run(_run(workdir, config, session_dir, resume))
    except KeyboardInterrupt:
        return 130


async def _run(
    workdir: Path,
    config: Config,
    session_dir: Path | None,
    resume: Path | None,
) -> int:
    workdir = workdir.resolve()
    reporters = create_reporters()
    install_exit_hooks(reporters)
    llm_client = create_llm_client(config)

    def make_agent(client: LLMClient) -> Agent:
        return Agent(
            config=config,
            llm_client=client,
            workdir=workdir,
            session_dir=session_dir,
        )

    def make_pump(agent: Agent) -> TurnPump:
        return TurnPump(
            agent,
            workdir,
            max_rounds=config.goal.max_rounds,
            verify_timeout_ms=config.goal.verify_timeout_ms,
        )

    agent = Agent(
        config=config,
        llm_client=llm_client,
        workdir=workdir,
        session_dir=session_dir,
        resume=resume,
    )

    def out(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    use_tty = os.name != "nt" and sys.stdin.isatty()
    inputs = (
        read_tty(sys.stdin.fileno(), out) if use_tty else read_pipe(sys.stdin)
    )
    frontend = HeadlessFrontend(
        config=config,
        agent=agent,
        pump=make_pump(agent),
        llm_client=llm_client,
        reporters=reporters,
        out=out,
        agent_factory=make_agent,
        pump_factory=make_pump,
        echo_input=not use_tty,
    )
    if use_tty:
        with _terminal_mode(sys.stdin.fileno()):
            return await frontend.run(inputs)
    return await frontend.run(inputs)
