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
(single-flight, steer drain, goal loop) stays in ``TurnPump``/``Agent``,
and session switching plus the /goal proposal flow in ``Conversation`` —
the frontend only translates stdin into conversation calls and pump
events into stdout text.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import re
import sys
import unicodedata
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from limbo.agent import (
    Agent,
    CompactionEvent,
    ErrorEvent,
    InterruptEvent,
    RetryEvent,
    SteerEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallRequest,
    ToolResultEvent,
    UsageUpdate,
)
from limbo.config import Config
from limbo.conversation import Conversation, NoVerifyGate
from limbo.daily import (
    DAILY_USAGE,
    daily_only_notice,
    daily_toggle_notice,
    daily_unchanged_notice,
)
from limbo.goal import (
    GoalClear,
    GoalQuery,
    GoalSet,
    parse_goal_args,
)
from limbo.integrations import create_reporters, install_exit_hooks
from limbo.integrations.base import IntegrationReporter
from limbo.llm.factory import create_llm_client
from limbo.memory import MEMORY_USAGE, describe, parse_memory_command
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
    QueueHeld,
    TurnPump,
)
from limbo.steer import FOLLOW_UP
from limbo.subagents.commands import agents_command, direct_message, task_command
from limbo.subagents.events import (
    BackgroundEvent,
    MainWake,
    NoticeEvent,
    PlanProposed,
    TaskChanged,
)
from limbo.toggles import parse_toggle
from limbo.tracing.command import TracingViewerCommand

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
class _Background:
    """Internal queue event: something happened on the task board."""

    event: BackgroundEvent


@dataclass(frozen=True)
class _TurnDone:
    """Internal queue event: the turn/compact worker task finished."""

    error: BaseException | None = None
    task: asyncio.Task[None] | None = None


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
class SlashQueue:
    """``/queue <text>``: a follow-up, handled after the current task."""

    text: str


@dataclass(frozen=True)
class SlashFork:
    """``/fork [n]``: branch into a new session (keep the first n turns)."""

    arg: str


@dataclass(frozen=True)
class SlashPlan:
    """``/plan [on|off]``: read-only plan mode."""

    arg: str


@dataclass(frozen=True)
class SlashDaily:
    """``/daily-mode [on|off]``: the daily (non-coding) posture."""

    arg: str


@dataclass(frozen=True)
class SlashLearn:
    """``/learn``: review the conversation into daily skills."""


@dataclass(frozen=True)
class SlashMemory:
    """``/memory [edit user|notes]``: the persistent memory files."""

    arg: str


@dataclass(frozen=True)
class SlashHandoff:
    """``/handoff <note>``: a linked new session that starts from the note."""

    note: str


@dataclass(frozen=True)
class SlashHelp:
    pass


@dataclass(frozen=True)
class SlashQuit:
    pass


@dataclass(frozen=True)
class SlashTracingViewer:
    arg: str


@dataclass(frozen=True)
class SlashAgents:
    """``/agents [scan|init]``: the task board / project profiles."""

    arg: str


@dataclass(frozen=True)
class SlashTask:
    """``/task ...``: direct control of subagent tasks."""

    arg: str


@dataclass(frozen=True)
class UnknownCommand:
    name: str


Submission = (
    Prompt
    | SlashGoal
    | SlashCompact
    | SlashModel
    | SlashNew
    | SlashQueue
    | SlashFork
    | SlashHandoff
    | SlashPlan
    | SlashDaily
    | SlashMemory
    | SlashLearn
    | SlashHelp
    | SlashQuit
    | SlashTracingViewer
    | SlashAgents
    | SlashTask
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
    if name == "queue":
        return SlashQueue(arg)
    if name == "fork":
        return SlashFork(arg)
    if name == "handoff":
        return SlashHandoff(arg)
    if name == "plan":
        return SlashPlan(arg)
    if name == "daily-mode":
        return SlashDaily(arg)
    if name == "memory":
        return SlashMemory(arg)
    if name == "learn":
        return SlashLearn()
    if name == "help":
        return SlashHelp()
    if name in ("quit", "exit"):
        return SlashQuit()
    if name == "tracing-viewer":
        return SlashTracingViewer(arg)
    if name == "agents":
        return SlashAgents(arg)
    if name == "task":
        return SlashTask(arg)
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

# Listed by /help only in daily mode (refused in coding sessions).
DAILY_HELP_TEXT = """
  /learn         复盘这次对话，把经验写进 daily skill（后台子代理）
  /memory        查看长期记忆（/memory edit [user|notes] 显示文件路径）"""

HELP_TEXT = """可用命令：
  /goal <目标>   设定目标并进入闭环（首轮提议验收命令）
  /goal clear    退出 goal 模式
  /compact       立即压缩对话历史
  /model [id]    查看/切换模型
  /new           开始新会话
  /queue <文本>  当前任务做完后再处理这条消息（空闲时直接发送）
  /fork [n]      分叉出新会话继续（n = 只保留前 n 轮），原会话不变
  /handoff <说明> 带交接说明开新会话（干净上下文，关联原会话）
  /plan [on|off] 计划模式：只读探索并给出方案，写类工具被拦下
  /daily-mode [on|off]  Daily mode：日常非 coding 工作（读网页、长期记忆）
  /tracing-viewer [id|off]  在浏览器查看 trace（打印本地链接）
  /agents [scan|init]  子代理任务板 · 项目画像 · 生成项目角色
  /task [id]     查看任务；/task <pause|resume|cancel|retry|approve|reject> <id|all> [说明]
  @子代理名 <消息>  直接对某个子代理说话（忙时也可用）
  /quit          退出
忙时输入的普通文本会作为插话排队；Esc / Ctrl+C 中断当前任务。
输入关闭后会等后台子代理做完（暂停、待确认的任务不等）再退出。"""


def _first_line(text: str, limit: int = 60) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


class HeadlessFrontend:
    """Drives one ``Conversation`` from a stream of InputEvents.

    Everything is serialized through one asyncio queue: the stdin reader
    (or a test's async iterator) feeds ``Submit``/``Interrupt``/``Eof``,
    and the turn worker's done-callback feeds ``_TurnDone`` — so goal
    post-turn logic, busy checks, and EOF handling never race.
    """

    def __init__(
        self,
        *,
        conversation: Conversation,
        reporters: IntegrationReporter,
        out: Callable[[str], None],
        echo_input: bool = False,
        max_tool_lines: int = 60,
        json_mode: bool = False,
    ) -> None:
        self._conv = conversation
        self._config = conversation.config
        # --json: every output line is one JSON object (events + info).
        self._json = json_mode
        self._reporters = reporters
        self._out = out
        self._echo_input = echo_input
        self._max_tool_lines = max_tool_lines

        self._queue: asyncio.Queue[InputEvent | _TurnDone | _Background] = asyncio.Queue()
        self._turn_task: asyncio.Task[None] | None = None
        self._eof = False
        self._eof_waiting = False
        self._quit = False
        # Restored from the session's usage ledger on resume.
        self._last_tokens = conversation.agent.session_total_tokens
        self._tracing_viewer = TracingViewerCommand(lambda: self._conv.agent)

    @property
    def _agent(self) -> Agent:
        return self._conv.agent

    @property
    def _pump(self) -> TurnPump:
        return self._conv.pump

    # -- main loop ------------------------------------------------------------

    async def run(self, inputs: AsyncIterator[InputEvent]) -> int:
        forwarder = asyncio.create_task(self._forward(inputs))
        unsubscribe = self._conv.subscribe(lambda e: self._queue.put_nowait(_Background(e)))
        self._conv.start_background()
        self._conv.report_session()
        self._conv.report_idle()
        if self._json:
            self._emit(
                {
                    "type": "ready",
                    "model": self._config.llm.model,
                    "workdir": str(self._agent.workdir),
                    "session_id": self._agent.session_id,
                    "total_tokens": self._last_tokens,
                }
            )
        else:
            self._line(
                f"limbo headless · model {self._config.llm.model} · "
                f"workdir {self._agent.workdir}"
            )
            self._line("输入消息回车发送；/help 查看命令；Esc/Ctrl+C 中断当前任务")
        sup = self._conv.supervisor
        if sup is not None and sup.interrupted_on_load:
            self._line(
                f"◆ {len(sup.interrupted_on_load)} 个子任务在上次退出时中断"
                f"（{'、'.join(sup.interrupted_on_load)}）；/task resume all 继续"
            )
        self._maybe_wake_main()
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
                elif isinstance(event, _Background):
                    self._on_background(event.event)
        finally:
            unsubscribe()
            forwarder.cancel()
            if self._turn_task is not None:
                self._turn_task.cancel()
            self._reporters.release()
            self._tracing_viewer.close()
            self._conv.close()
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
        if self._conv.awaiting_verify:
            self._answer_verify(text)
            return
        if text.startswith("@") and self._direct_message(text):
            return
        command = parse_submission(text)
        if isinstance(command, (SlashAgents, SlashTask)):
            # Board commands never touch the main history: fine mid-turn.
            self._cmd_board(command)
            return
        # The pump's own flag (not "a worker task exists"): it is released
        # in the same await-free section as the pump's last drain, so a
        # submission in the teardown window starts a turn instead of being
        # steered into a queue nobody drains any more.
        if self._conv.busy:
            # Same discipline as the TUI (RFC LIM-20): plain text steers,
            # history-shaping commands are refused while a turn owns the
            # agent; /help stays available.
            if isinstance(command, Prompt):
                self._agent.steer(text)
                self._line(f"➤ 已排队插话：{_first_line(text)}")
            elif isinstance(command, SlashQueue) and command.text:
                self._agent.steer(command.text, mode=FOLLOW_UP)
                self._line(f"➤ 已排队，当前任务完成后处理：{_first_line(command.text)}")
            elif isinstance(command, SlashHelp):
                self._line(HELP_TEXT)
            elif isinstance(command, SlashTracingViewer):
                await self._cmd_tracing_viewer(command.arg)
            else:
                self._line(
                    "当前任务进行中，该命令暂不可用"
                    "（Esc/Ctrl+C 中断，或等待完成）"
                )
            return
        if isinstance(command, Prompt):
            self._start(self._pump.run(text))
        elif isinstance(command, SlashQueue):
            if command.text:
                # Idle: nothing to wait for, so it is a plain prompt.
                self._start(self._pump.run(command.text))
            else:
                self._line("用法：/queue <文本>")
        elif isinstance(command, SlashGoal):
            self._cmd_goal(command.arg)
        elif isinstance(command, SlashCompact):
            self._start(self._pump.compact())
        elif isinstance(command, SlashModel):
            await self._cmd_model(command.arg)
        elif isinstance(command, SlashNew):
            self._cmd_new()
        elif isinstance(command, SlashFork):
            self._cmd_fork(command.arg)
        elif isinstance(command, SlashHandoff):
            self._cmd_handoff(command.note)
        elif isinstance(command, SlashPlan):
            self._cmd_plan(command.arg)
        elif isinstance(command, SlashDaily):
            self._cmd_daily(command.arg)
        elif isinstance(command, SlashMemory | SlashLearn) and not self._agent.daily_mode:
            self._line(daily_only_notice("/memory" if isinstance(command, SlashMemory)
                                         else "/learn"))
        elif isinstance(command, SlashMemory):
            self._cmd_memory(command.arg)
        elif isinstance(command, SlashLearn):
            self._line(self._conv.learn())
        elif isinstance(command, SlashHelp):
            self._line(HELP_TEXT + (DAILY_HELP_TEXT if self._agent.daily_mode else ""))
        elif isinstance(command, SlashQuit):
            self._quit = True
        elif isinstance(command, SlashTracingViewer):
            await self._cmd_tracing_viewer(command.arg)
        elif isinstance(command, UnknownCommand):
            self._line(f"未知命令 /{command.name}（/help 查看可用命令）")

    def _cmd_plan(self, arg: str) -> None:
        enabled = parse_toggle(arg, self._agent.plan_mode)
        if enabled is None:
            self._line("用法：/plan on|off（无参则切换）")
        elif self._agent.set_plan_mode(enabled):
            self._line(
                "计划模式已开启：只读探索，edit/write 和会改动状态的命令会被拦下"
                if enabled
                else "计划模式已关闭：工具恢复正常"
            )
        else:
            self._line(f"计划模式已是{'开启' if enabled else '关闭'}状态")

    def _cmd_daily(self, arg: str) -> None:
        enabled = parse_toggle(arg, self._agent.daily_mode)
        if enabled is None:
            self._line(DAILY_USAGE)
        elif self._agent.set_daily_mode(enabled):
            self._line(daily_toggle_notice(enabled))
        else:
            self._line(daily_unchanged_notice(enabled))

    def _cmd_memory(self, arg: str) -> None:
        try:
            target = parse_memory_command(arg)
        except ValueError:
            self._line(MEMORY_USAGE)
            return
        store = self._agent.registry.memory_store
        if target is None or store is None:
            self._line(describe(store, self._agent.memory_snapshot))
        else:
            # No editor here: the REPL owns the terminal (raw mode).
            self._line(f"直接编辑 {store.path(target)}，从下一个会话起生效")

    def _on_interrupt(self, event: Interrupt) -> None:
        if self._conv.awaiting_verify:
            # Esc/Ctrl+C on the proposal prompt = decline (picker cancel).
            self._conv.skip_verify()
            self._line("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            return
        if self._conv.busy:
            self._agent.interrupt()
            self._pump.cancel_verify()
            return
        if event.hard:  # Ctrl+C at idle quits; Esc at idle is a no-op.
            self._quit = True

    def _on_eof(self) -> None:
        self._eof = True
        if self._conv.awaiting_verify:
            self._skip_verify_on_eof()
            self._quit = True
            return
        if self._turn_task is not None:
            self._line("输入已关闭，等待当前任务完成…")
            return
        self._quit_when_settled()

    def _quit_when_settled(self) -> None:
        """EOF: quit once no turn runs and background subagents are done."""
        sup = self._conv.supervisor
        if self._turn_task is None and (sup is None or not sup.working):
            self._quit = True
        elif self._turn_task is None and not self._eof_waiting:
            self._eof_waiting = True
            self._line("输入已关闭，等待后台子代理完成…")

    def _skip_verify_on_eof(self) -> None:
        self._conv.skip_verify()
        self._line("输入已关闭，跳过验收确认（保持单轮模式）")

    # -- turn worker ----------------------------------------------------------

    def _start(self, events: AsyncIterator[PumpEvent]) -> None:
        # pump.run()/compact() set the busy flag eagerly at call time, so
        # single-flight holds even before the task first runs.
        task = asyncio.create_task(self._pump_events(events))
        self._turn_task = task
        task.add_done_callback(
            lambda t: self._queue.put_nowait(
                _TurnDone(None if t.cancelled() else t.exception(), t)
            )
        )

    async def _pump_events(self, events: AsyncIterator[PumpEvent]) -> None:
        try:
            await self._conv.drive(events, self._render)
        except Exception as e:  # noqa: BLE001 - mirror the TUI: surface it
            self._line(f"✗ 内部错误：{e}")
        finally:
            if self._json:
                self._emit({"type": "idle", "total_tokens": self._last_tokens})
            else:
                self._line(f"── limbo idle · 累计 tokens {self._last_tokens} ──")

    def _on_turn_done(self, done: _TurnDone) -> None:
        if done.task is not self._turn_task:
            return  # a newer turn started in the teardown window
        self._turn_task = None
        if done.error is not None:
            self._line(f"✗ 内部错误：{done.error}")
        self._maybe_confirm_verify_proposal()
        if self._eof and self._conv.awaiting_verify:
            self._skip_verify_on_eof()
            self._quit = True
            return
        self._maybe_wake_main()
        if self._eof:
            self._quit_when_settled()

    # -- subagents ------------------------------------------------------------

    def _cmd_board(self, command: SlashAgents | SlashTask) -> None:
        sup = self._conv.supervisor
        if sup is None:
            self._line('子代理已关闭（config.toml 的 [subagents] auto = "off"）')
        elif isinstance(command, SlashAgents):
            self._line(agents_command(sup, command.arg))
        else:
            self._line(task_command(sup, command.arg))

    def _direct_message(self, text: str) -> bool:
        sup = self._conv.supervisor
        reply = direct_message(sup, text) if sup is not None else None
        if reply is None:
            return False
        self._line(f"➤ {reply}")
        return True

    def _maybe_wake_main(self) -> None:
        """Start a main-agent turn for subagent notices queued while idle."""
        if self._turn_task is not None or self._conv.awaiting_verify or self._quit:
            return
        text = self._conv.take_wake_input()
        if text is None:
            return
        self._line("◆ 子代理有新进展，交给主 agent 处理")
        self._start(self._pump.run(text))

    def _on_background(self, event: BackgroundEvent) -> None:
        if self._json:
            self._emit(background_to_json(event))
        elif isinstance(event, NoticeEvent):
            notice = event.notice
            if notice.headline and notice.task_id:
                icon = {"result": "✓", "alert": "⚠", "question": "?"}.get(notice.kind, "◆")
                self._line(f"{icon} {notice.headline}")
        elif isinstance(event, PlanProposed):
            sup = self._conv.supervisor
            if sup is not None:
                self._line("◆ 子任务计划待确认：")
                self._line(_indent(sup.status_text()))
                self._line("  /task approve all 开始 · /task reject all 放弃")
        if isinstance(event, MainWake):
            self._maybe_wake_main()
        if self._eof and not self._quit:
            self._quit_when_settled()

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
            if self._conv.clear_goal():
                self._line("已退出 goal 模式")
            else:
                self._line("当前没有 goal")
            return
        # GoalSet: the first round is a proposal round; the confirmation
        # happens when that turn ends (see _maybe_confirm_verify_proposal).
        assert isinstance(command, GoalSet)
        self._start(self._pump.run(self._conv.start_goal(command.text)))

    def _maybe_confirm_verify_proposal(self) -> None:
        """After a proposal round, present the model's verify command(s).

        The conversation parses the proposal and reports ``blocked``, which
        is what ``herdr agent wait --until blocked`` syncs on; the next
        submission is the answer.
        """
        proposal = self._conv.take_verify_proposal()
        if proposal is None:
            return
        if isinstance(proposal, NoVerifyGate):
            self._line(
                "模型认为该目标没有客观可判定的验收方式，保持单轮模式；"
                "/goal clear 退出"
            )
            return
        if proposal.commands:
            self._line("模型提议的验收命令：")
            for index, command in enumerate(proposal.commands, 1):
                self._line(f"  {index}. {command}")
        else:
            self._line("未能解析模型的验收提议。")
        self._line("回复序号采纳；或直接输入一行作为验收命令；/skip 跳过自动验收")

    def _answer_verify(self, text: str) -> None:
        answer = parse_verify_answer(text, self._conv.proposals)
        if isinstance(answer, InvalidVerifyAnswer):
            self._line(answer.reason)
            return
        if isinstance(answer, SkipVerify):
            self._conv.skip_verify()
            self._line("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            return
        prompt = self._conv.accept_verify(answer.command)
        if prompt is None:
            return
        self._line(f"验收方式已确认：{answer.command}")
        self._start(self._pump.run(prompt))

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
        self._conv.llm_client, notices = await swap_llm_client(
            self._config, self._conv.llm_client, self._agent
        )
        for notice in notices:
            self._line(notice)

    async def _cmd_tracing_viewer(self, arg: str) -> None:
        self._line(await asyncio.to_thread(self._tracing_viewer.run, arg))

    def _cmd_new(self) -> None:
        self._conv.new()
        self._last_tokens = 0
        self._line("已开始新会话")

    def _cmd_fork(self, arg: str) -> None:
        if arg and not arg.isdigit():
            self._line("用法：/fork [n]（n = 保留前 n 轮对话，省略则全部保留）")
            return
        parent = self._conv.fork(int(arg) if arg else None)
        self._last_tokens = self._agent.session_total_tokens
        self._line(f"已从会话 {parent} 分叉，新会话 {self._agent.session_id}")

    def _cmd_handoff(self, note: str) -> None:
        if not note:
            self._line("用法：/handoff <交接说明>")
            return
        parent, prompt = self._conv.handoff(note)
        self._last_tokens = self._agent.session_total_tokens
        self._line(f"已交接：新会话 {self._agent.session_id}，关联原会话 {parent}")
        self._start(self._pump.run(prompt))

    # -- rendering ------------------------------------------------------------

    def _line(self, text: str = "") -> None:
        if self._json:
            self._emit({"type": "info", "text": text})
        else:
            self._out(text + "\n")

    def _emit(self, record: dict[str, Any]) -> None:
        self._out(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")

    def _render(self, event: PumpEvent) -> None:
        if self._json:
            if isinstance(event, UsageUpdate):
                self._last_tokens = event.total_tokens
            self._emit(event_to_json(event))
            return
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
        elif isinstance(event, RetryEvent):
            self._line(
                f"\n⟳ 连接中断，重试（{event.attempt}/{event.max_attempts}）："
                f"{event.reason}；以上半段回复作废"
            )
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
        elif isinstance(event, QueueHeld):
            count = len(event.items)
            if event.policy == "withdraw":
                texts = "；".join(_first_line(item.text) for item in event.items)
                self._line(f"上一轮出错，已撤回 {count} 条排队消息：{texts}")
            else:
                self._line(f"上一轮出错，{count} 条排队消息保留，下次发送时一起处理")


# -- --json event stream ------------------------------------------------------------


def background_to_json(event: BackgroundEvent) -> dict[str, Any]:
    """A task-board event as one ``--json`` line."""
    if isinstance(event, TaskChanged):
        task = event.task
        return {
            "type": "task",
            "id": task.id,
            "status": task.status,
            "title": task.title,
            "agent": task.agent_name,
            "activity": event.activity,
            "waiting_on": task.waiting_on,
            "checkpoint": task.last_checkpoint.phase if task.last_checkpoint else None,
        }
    if isinstance(event, NoticeEvent):
        return {
            "type": "subagent_notice",
            "id": event.notice.id,
            "kind": event.notice.kind,
            "task_id": event.notice.task_id,
            "text": event.notice.headline,
        }
    if isinstance(event, PlanProposed):
        return {"type": "plan_proposed", "task_ids": event.task_ids}
    return {"type": "main_wake"}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def event_to_json(event: PumpEvent) -> dict[str, Any]:
    """One agent/goal event as a JSON-ready dict: ``type`` + its fields."""
    record: dict[str, Any] = {"type": _snake(type(event).__name__)}
    if dataclasses.is_dataclass(event):
        for field in dataclasses.fields(event):
            record[field.name] = getattr(event, field.name)
    return record


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


# -- entry point ------------------------------------------------------------------


def run_headless(
    *,
    workdir: Path,
    config: Config,
    session_dir: Path | None,
    resume: Path | None,
    json_mode: bool = False,
    daily: bool = False,
) -> int:
    """Synchronous entry for ``limbo --headless`` (called from app.main)."""
    try:
        return asyncio.run(_run(workdir, config, session_dir, resume, json_mode, daily))
    except KeyboardInterrupt:
        return 130


async def _run(
    workdir: Path,
    config: Config,
    session_dir: Path | None,
    resume: Path | None,
    json_mode: bool = False,
    daily: bool = False,
) -> int:
    workdir = workdir.resolve()
    def out(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    # OSC 7501 program status shares stdout with the transcript, so never
    # in --json mode (it would corrupt the JSON lines) or into a pipe.
    terminal = out if not json_mode and sys.stdout.isatty() else None
    reporters = create_reporters(terminal=terminal)
    install_exit_hooks(reporters)
    conversation = Conversation(
        config=config,
        workdir=workdir,
        llm_client=create_llm_client(config),
        session_dir=session_dir,
        resume=resume,
        reporters=reporters,
        client_factory=create_llm_client,
        daily=daily,
    )

    # --json is for scripts: plain line reads, no raw-mode echo.
    use_tty = not json_mode and os.name != "nt" and sys.stdin.isatty()
    inputs = (
        read_tty(sys.stdin.fileno(), out) if use_tty else read_pipe(sys.stdin)
    )
    frontend = HeadlessFrontend(
        conversation=conversation,
        reporters=reporters,
        out=out,
        echo_input=not use_tty and not json_mode,
        json_mode=json_mode,
    )
    if use_tty:
        with _terminal_mode(sys.stdin.fileno()):
            return await frontend.run(inputs)
    return await frontend.run(inputs)
