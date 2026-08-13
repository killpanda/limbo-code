"""Main screen: pi-style single-column chat layout."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

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
from limbo.compaction import is_summary_message
from limbo.config import Config
from limbo.goal import (
    STATUS_ACTIVE,
    GoalClear,
    GoalQuery,
    build_initial_prompt,
    build_proposer_prompt,
    parse_goal_args,
    parse_verify_proposal,
)
from limbo.integrations import AgentState, create_reporters, install_exit_hooks
from limbo.llm.client import LLMClient
from limbo.llm.factory import create_llm_client
from limbo.model_switch import (
    prepare_model_switch,
    reload_llm_config,
    swap_llm_client,
)
from limbo.models import Attachment
from limbo.pump import (
    GoalExhausted,
    GoalPassed,
    GoalResumed,
    GoalVerifyResultEvent,
    GoalVerifyStarted,
    PumpEvent,
    TurnPump,
)
from limbo.sessions import derive_title, export_jsonl, export_markdown, list_sessions
from limbo.skills import Skill, discover_skills
from limbo.ui.banner import startup_art_text
from limbo.ui.commands import SlashCommand, SlashCommandRegistry
from limbo.ui.path_input import attachments_from_text, looks_like_path
from limbo.ui.screens.btw import BtwScreen
from limbo.ui.screens.game2048 import Game2048Screen
from limbo.ui.screens.model_picker import ModelPicker
from limbo.ui.screens.session_picker import SessionPicker
from limbo.ui.screens.verify_picker import EDIT as VERIFY_EDIT
from limbo.ui.screens.verify_picker import VerifyPicker
from limbo.ui.widgets.chat import ChatWidget, QueuedMessage
from limbo.ui.widgets.command_menu import SlashCommandMenu
from limbo.ui.widgets.input import InputWidget, PasteMarkersInvalid, UserSubmitted
from limbo.ui.widgets.status_bar import StatusBar


class MainScreen(Screen[None]):
    """Single-column chat screen: status bar / chat flow / input / hint."""

    BINDINGS = [
        Binding("ctrl+o", "toggle_tools", "展开/收起工具输出"),
        Binding("ctrl+g", "game2048", "2048 小游戏"),
        # Screen-level ESC (LIM-53): the input widget's priority binding
        # wins while it is focused (slash-menu close lives there); this
        # binding makes ESC work when focus is elsewhere (e.g. the chat
        # was clicked). Modal screens on top handle ESC in their own
        # bindings before this screen ever sees it.
        Binding("escape", "handle_escape", show=False),
    ]

    def __init__(
        self,
        workdir: Path,
        config: Config | None = None,
        llm_client: LLMClient | None = None,
        session_dir: Path | None = None,
        resume: Path | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workdir = workdir
        self.config = config or Config()
        self.llm_client = llm_client or create_llm_client(self.config)
        self.session_dir = session_dir or Path.home() / ".limbo" / "sessions"
        self.agent = self._new_agent(resume=resume)
        self.pump = self._new_pump(self.agent)
        self._slash_menu_open = False
        # M2: proposal-round bookkeeping. _awaiting_verify_proposal is set
        # when /goal starts a proposal round; the confirmation picker pops
        # at the next quiet turn boundary. _editing_verify marks that the
        # next submission is a verify command being edited (not a message).
        self._awaiting_verify_proposal = False
        self._editing_verify = False
        self._pending_proposals: list[str] = []
        self._commands = SlashCommandRegistry()
        self._register_builtin_commands()
        # External tool integrations (Herdr, ...): an empty composite
        # outside any integrated environment, so all calls are no-ops.
        self._integrations = create_reporters()
        install_exit_hooks(self._integrations)

    @property
    def _agent_busy(self) -> bool:
        """Busy while the turn pump owns the agent history.

        One flag covers everything (turns, goal rounds, follow-ups and
        /compact mini-turns): ``TurnPump.running``, set eagerly when
        ``pump.run()``/``pump.compact()`` is called.
        """
        return self.pump.running

    def _new_pump(self, agent: Agent) -> TurnPump:
        return TurnPump(
            agent,
            self.workdir,
            max_rounds=self.config.goal.max_rounds,
            verify_timeout_ms=self.config.goal.verify_timeout_ms,
        )

    def _new_agent(self, resume: Path | None = None) -> Agent:
        return Agent(
            config=self.config,
            llm_client=self.llm_client,
            workdir=self.workdir,
            session_dir=self.session_dir,
            resume=resume,
        )

    # -- External tool integrations (no-op outside integrated environments) --

    def _report_state(
        self, state: AgentState, *, message: str | None = None
    ) -> None:
        """Report semantic lifecycle state to active integrations."""
        self._integrations.report(state, message=message)

    def _report_session(self) -> None:
        """Report the current session identity (startup, /new, /sessions)."""
        self._integrations.report_session(
            self.agent.session_id, str(self.agent.session_path)
        )

    def compose(self) -> ComposeResult:
        yield StatusBar(
            model=self.config.llm.model,
            workdir=str(self.workdir),
            id="statusbar",
        )
        yield ChatWidget(id="chat")
        yield SlashCommandMenu(id="slash-menu")
        yield InputWidget(id="input")
        yield Static(
            "Enter 发送 · Shift+Enter 换行 · ↑↓ 历史输入 · / 命令 · ctrl+o 工具输出 · ctrl+g 2048",
            id="hint",
            markup=False,
        )

    def on_mount(self) -> None:
        chat = self.query_one("#chat", ChatWidget)
        resumed = len(self.agent.messages) > 1
        if not resumed and self.config.ui.show_banner:
            chat.add_art(startup_art_text())
        chat.add_info(f"Limbo ready · {self.config.llm.model} · {self.workdir}")
        if resumed:
            self._render_history()
            meta = self.agent.session_meta
            chat.add_info(
                f"已恢复会话 {meta.id} · {meta.title or '(无标题)'}"
            )
        self._refresh_goal_indicator()  # D6: silent resume, badge only
        self.query_one("#input", InputWidget).focus()
        self._report_session()
        self._report_state("idle")

    # -- slash command menu ---------------------------------------------------

    @property
    def slash_menu_open(self) -> bool:
        return self._slash_menu_open

    def on_text_area_changed(self, event) -> None:
        """Show/filter the command menu while the input starts with '/'."""
        if getattr(event.text_area, "id", None) != "input":
            return
        text = event.text_area.text
        if text.startswith("/") and not any(ch.isspace() for ch in text):
            matches = [
                c for c in self._slash_candidates() if c.name.startswith(text)
            ]
            menu = self.query_one("#slash-menu", SlashCommandMenu)
            if matches:
                menu.show_commands(matches)
                self._slash_menu_open = True
                return
            # No command matches: show a passive hint instead of letting the
            # user discover the outcome on submit. The menu-open flag stays
            # False, so Enter/arrows/Esc keep their normal behavior.
            if Path(text).expanduser().is_file():
                menu.show_hint("📎 路径已识别 · 回车将作为附件发送")
            elif looks_like_path(text):
                menu.show_hint("无匹配命令 · 将作为普通文本发送")
            else:
                menu.show_hint("无匹配命令 · /help 查看全部命令")
            self._slash_menu_open = False
            return
        self.slash_menu_close()

    def _slash_candidates(self) -> list:
        """Built-in commands plus discovered skills. Re-scanned on each menu
        update so skills added while Limbo is running appear immediately."""
        return self._commands.candidates(discover_skills(self.workdir))

    def slash_menu_move(self, delta: int) -> None:
        menu = self.query_one("#slash-menu", SlashCommandMenu)
        if delta > 0:
            menu.action_cursor_down()
        else:
            menu.action_cursor_up()

    def slash_menu_close(self) -> None:
        self._slash_menu_open = False
        self.query_one("#slash-menu", SlashCommandMenu).close()

    def slash_menu_complete(self, execute: bool) -> bool:
        """Complete the highlighted command. Returns False if nothing done.

        With ``execute=True``, commands that take no arguments run
        immediately; arg-taking commands are completed into the input so the
        user can type the argument.
        """
        if not self._slash_menu_open:
            return False
        menu = self.query_one("#slash-menu", SlashCommandMenu)
        command = menu.highlighted_command()
        if command is None:
            return False
        input_widget = self.query_one("#input", InputWidget)
        if execute and input_widget.text.strip() == command.name:
            # Exact match: Enter submits the command as typed (e.g. invoking
            # a skill without args) instead of completing it.
            self.slash_menu_close()
            return False
        self.slash_menu_close()
        if execute and not command.takes_args:
            input_widget.clear()
            self._handle_command(command.name)
        else:
            input_widget.text = command.name + " "
            input_widget.move_cursor(input_widget.document.end)
            input_widget.focus()
        return True

    def on_option_list_option_selected(self, event) -> None:
        """Mouse click on a menu item completes it like Enter."""
        if isinstance(getattr(event, "option_list", None), SlashCommandMenu):
            event.stop()
            self.slash_menu_complete(execute=True)

    # -- slash commands ---------------------------------------------------------

    # -- /goal: closed-loop goal mode (LIM-40) -------------------------------

    def _goal(self, arg: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        command = parse_goal_args(arg)
        if isinstance(command, GoalQuery):
            state = self.pump.status()
            if state is None or state.status == "cleared":
                chat.add_info("当前没有 goal。用法：/goal <目标> · /goal clear")
            else:
                chat.add_info(
                    f"🎯 Goal（{state.status} · 已验收 "
                    f"{state.rounds_completed}/{state.max_rounds} 轮）\n"
                    f"目标：{state.text}\n"
                    f"验收命令：{state.verify_command or '（未设置）'}"
                )
            return
        if isinstance(command, GoalClear):
            if self.pump.status() is None:
                chat.add_info("当前没有 goal")
                return
            self.pump.clear()
            self._awaiting_verify_proposal = False
            self._editing_verify = False
            suffix = "（当前轮跑完后停止续轮）" if self.pump.running else ""
            chat.add_info(f"已退出 goal 模式{suffix}")
            self._refresh_goal_indicator()
            return
        # GoalSet: starting a new goal is a history-shaping action, so it
        # follows the same busy rule as other rewriting commands (LIM-20).
        if self._agent_busy:
            chat.add_info("当前任务进行中，请等待完成后再设定 /goal")
            return
        state = self.pump.set_goal(command.text)
        chat.add_user_message(f"/goal {command.text}")
        self._refresh_goal_indicator()
        # M2: the first round is a proposal round — the model explores the
        # repo and proposes the acceptance command(s); the confirmation
        # picker pops when the turn ends (see _maybe_confirm_verify_proposal).
        self._awaiting_verify_proposal = True
        self.run_worker(self._handle_turn(build_proposer_prompt(state)))

    def _refresh_goal_indicator(self) -> None:
        """Sync the status-bar badge with pump state (rainbow = running)."""
        statusbar = self.query_one("#statusbar", StatusBar)
        state = self.pump.status()
        if state is not None and state.status == STATUS_ACTIVE:
            statusbar.set_goal(
                (state.rounds_completed, state.max_rounds),
                running=self.pump.running,
            )
        else:
            statusbar.set_goal(None)

    # -- M2: model-proposed verify command confirmation ------------------------

    def _maybe_confirm_verify_proposal(self) -> None:
        """After a proposal round, confirm the model's verify command.

        Only fires at a quiet turn boundary (the finally-chain handles
        leftover steers first, per LIM-20): parse the last assistant
        message for <verify_proposal> and pop the picker.
        """
        if not self._awaiting_verify_proposal:
            return
        state = self.pump.status()
        if state is None or state.status != STATUS_ACTIVE or state.verify_command:
            self._awaiting_verify_proposal = False
            return
        text = ""
        for msg in reversed(self.agent.messages):
            if msg.role == "assistant" and msg.content:
                text = msg.content
                break
        parsed = parse_verify_proposal(text)
        chat = self.query_one("#chat", ChatWidget)
        if parsed is not None and len(parsed) == 0:
            # Explicit <none/>: no objective gate exists for this goal.
            self._awaiting_verify_proposal = False
            chat.add_info(
                "模型认为该目标没有客观可判定的验收方式，保持单轮模式；"
                "/goal clear 退出"
            )
            return
        self._pending_proposals = parsed or []
        if not self._pending_proposals:
            chat.add_info("未能解析模型的验收提议，请手动输入或跳过")
        self._report_state("blocked", message="等待确认 /goal 验收命令")
        self.app.push_screen(
            VerifyPicker(self._pending_proposals), self._on_verify_picked
        )

    def _on_verify_picked(self, choice: str | None) -> None:
        chat = self.query_one("#chat", ChatWidget)
        self._awaiting_verify_proposal = False
        state = self.pump.status()
        if state is None or state.status != STATUS_ACTIVE:
            return
        if choice is None:
            chat.add_info("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            self._refresh_goal_indicator()
            self._report_state("idle")
            return
        if choice == VERIFY_EDIT:
            # Prefill the best proposal for editing; the next submission is
            # treated as the verify command (see on_user_submitted).
            input_widget = self.query_one("#input", InputWidget)
            prefill = self._pending_proposals[0] if self._pending_proposals else ""
            input_widget.text = prefill
            input_widget.move_cursor(input_widget.document.end)
            input_widget.focus()
            self._editing_verify = True
            chat.add_info("编辑验收命令后回车确认（清空回车则跳过）")
            self._report_state("blocked", message="等待编辑 /goal 验收命令")
            return
        self._accept_verify(choice)

    def _accept_verify(self, command: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        state = self.pump.set_verify(command)
        if state is None:
            return
        chat.add_info(f"验收方式已确认：{command}")
        self._refresh_goal_indicator()
        self.run_worker(self._handle_turn(build_initial_prompt(state)))

    def _register_builtin_commands(self) -> None:
        self._commands.register(
            SlashCommand(
                "/sessions", "切换历史会话", handler=lambda arg: self._open_session_picker()
            )
        )
        self._commands.register(
            SlashCommand("/new", "开始新会话", handler=lambda arg: self._start_new_session())
        )
        self._commands.register(
            SlashCommand(
                "/export",
                "导出会话日志（默认 JSONL，路径以 .md 结尾则导出 Markdown）[path]",
                takes_args=True,
                handler=lambda arg: self._export_session(arg),
            )
        )
        self._commands.register(
            SlashCommand(
                "/compact",
                "立即压缩对话历史（释放上下文）",
                handler=lambda arg: self._compact_now(),
            )
        )
        self._commands.register(
            SlashCommand(
                "/model",
                "切换模型（无参打开选择器）[model]",
                takes_args=True,
                handler=lambda arg: self._switch_model(arg),
            )
        )
        self._commands.register(
            SlashCommand(
                "/help",
                "显示帮助",
                allow_when_busy=True,
                handler=lambda arg: self._show_help(),
            )
        )
        self._commands.register(
            SlashCommand(
                "/btw",
                "侧问：不打断当前任务的快速提问（不进会话历史）[问题]",
                takes_args=True,
                allow_when_busy=True,
                handler=lambda arg: self._btw(arg),
            )
        )
        self._commands.register(
            SlashCommand(
                "/2048",
                "玩一局 2048（不打断当前任务）",
                allow_when_busy=True,
                handler=lambda arg: self._open_game2048(),
            )
        )
        self._commands.register(
            SlashCommand(
                "/goal",
                "闭环目标模式：/goal <目标>（模型自拟验收，确认后自动闭环）· /goal clear",
                takes_args=True,
                # Query/clear must work mid-turn; the set variant re-checks
                # busy inside the handler.
                allow_when_busy=True,
                handler=lambda arg: self._goal(arg),
            )
        )

    def _handle_command(self, text: str) -> bool:
        """Dispatch a slash command; False means "treat as a normal message".

        False is returned only for '/'-leading text that is neither a known
        command nor a skill but looks like a file path — the caller then
        sends it as a plain message (auto-attaching existing files).
        Everything else (executed, busy-rejected, unknown-command error)
        returns True.
        """
        chat = self.query_one("#chat", ChatWidget)
        name, _, arg = text.partition(" ")
        arg = arg.strip()

        resolved = self._commands.resolve(name, discover_skills(self.workdir))
        if isinstance(resolved, SlashCommand) and resolved.handler is not None:
            # Busy guard (RFC LIM-20): history-rewriting commands are
            # rejected mid-turn. Centralized here so both entry points
            # (on_user_submitted and slash_menu_complete) are covered.
            if self._agent_busy and not resolved.allow_when_busy:
                self._reject_busy_command(resolved)
                return True
            resolved.handler(arg)
            return True
        if isinstance(resolved, Skill):
            self._invoke_skill(resolved, arg)
            return True
        if looks_like_path(text):
            return False
        chat.add_info(
            f"未知命令 {name} · /help 查看全部命令；"
            "若想发送以 / 开头的文本，请在前面加一个空格"
        )
        return True

    def _reject_busy_command(self, command: SlashCommand) -> None:
        chat = self.query_one("#chat", ChatWidget)
        if command.name == "/compact":
            # Keep the pre-existing wording for /compact.
            chat.add_info("当前任务进行中，请等待完成后再压缩")
        else:
            chat.add_info(f"当前任务进行中，请等待完成后再执行 {command.name}")

    def _compact_now(self) -> None:
        """Run /compact: the busy guard lives in _handle_command (RFC LIM-20);
        here we just pump the agent's compaction events like a mini-turn."""
        self.run_worker(self._run_compact())

    async def _run_compact(self) -> None:
        input_widget = self.query_one("#input", InputWidget)
        statusbar = self.query_one("#statusbar", StatusBar)
        # The pump owns the busy discipline (pump.compact() sets the shared
        # running flag eagerly): a concurrent turn or second /compact can
        # never race compact()'s wholesale history rewrite. What remains
        # here is pure UI: disable input, label the status bar.
        input_widget.disabled = True
        statusbar.set_state("compacting…", "thinking")
        self._report_state("working")
        try:
            async for event in self.pump.compact():
                await self._process_agent_event(event)
        finally:
            input_widget.disabled = False
            input_widget.focus()
            statusbar.set_state("idle")
            self._report_state("idle")

    def _show_help(self) -> None:
        self.query_one("#chat", ChatWidget).add_info(self._commands.help_text())

    # -- /model: runtime model switching --------------------------------------

    def _switch_model(self, arg: str) -> None:
        """Handle /model: open the picker, or switch directly with an arg."""
        chat = self.query_one("#chat", ChatWidget)
        if self._agent_busy:
            # Same discipline as /compact: swapping the client mid-stream
            # would corrupt the in-flight turn.
            chat.add_info("当前任务进行中，请等待完成后再切换模型")
            return
        self._reload_llm_config()
        if arg:
            self._apply_model_switch(arg)
            return
        self.app.push_screen(
            ModelPicker(self.config, self.config.llm.model),
            self._on_model_picked,
        )

    def _on_model_picked(self, model_id: str | None) -> None:
        if model_id is not None:
            self._apply_model_switch(model_id)

    def _reload_llm_config(self) -> None:
        """Hot-reload llm/providers settings (see limbo.model_switch)."""
        reload_llm_config(self.config)

    def _apply_model_switch(self, model_id: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        verdict = prepare_model_switch(model_id, self.config)
        for notice in verdict.notices:
            chat.add_info(notice)
        if verdict.switched:
            self.run_worker(self._swap_llm_client())

    async def _swap_llm_client(self) -> None:
        """Swap the client for the *current* config model (converges on
        the latest model when /model fires rapidly — see model_switch)."""
        chat = self.query_one("#chat", ChatWidget)
        model_id = self.config.llm.model
        self.llm_client, notices = await swap_llm_client(
            self.config, self.llm_client, self.agent
        )
        self.query_one("#statusbar", StatusBar).set_model(model_id)
        for notice in notices:
            chat.add_info(notice)

    def _start_new_session(self) -> None:
        chat = self.query_one("#chat", ChatWidget)
        self.agent.close()  # release the old session's trace file handle
        self.agent = self._new_agent()
        # The pump wraps the agent: replacing the agent without rebuilding
        # the pump would keep pumping the OLD agent's history (pre-existing
        # /new bug — the resume path below always did both).
        self.pump = self._new_pump(self.agent)
        chat.clear()
        chat.add_info(f"已开始新会话 {self.agent.session_id}")
        self._report_session()

    def _invoke_skill(self, skill: Skill, arg: str) -> None:
        """Invoke a skill: its body becomes the turn's instruction.

        Busy (RFC LIM-20): the assembled prompt joins the steer queue
        instead of starting a second concurrent turn worker.
        """
        display = f"/{skill.name}" + (f" {arg}" if arg else "")
        prompt = (
            f"# Skill: {skill.name}\n\n{skill.body.strip()}\n\n"
            f"(Skill 文件位于 {skill.path}，其中引用的相对路径基于其所在目录解析。)"
        )
        if arg:
            prompt += f"\n\n## 用户输入\n\n{arg}"
        chat = self.query_one("#chat", ChatWidget)
        if self._agent_busy:
            item_id = self.agent.steer(prompt)
            chat.add_queued_message(item_id, display)
            self._update_queue_status()
            return
        chat.add_user_message(display)
        self.run_worker(self._handle_turn(prompt))

    def _open_session_picker(self) -> None:
        chat = self.query_one("#chat", ChatWidget)
        sessions = list_sessions(self.session_dir, workdir=self.workdir)
        if not sessions:
            chat.add_info("没有可切换的历史会话")
            return
        self.app.push_screen(SessionPicker(sessions), self._on_session_picked)

    def _on_session_picked(self, path: Path | None) -> None:
        if path is None:
            return
        chat = self.query_one("#chat", ChatWidget)
        self.agent.close()  # release the old session's trace file handle
        self.agent = self._new_agent(resume=path)
        self.pump = self._new_pump(self.agent)
        self._awaiting_verify_proposal = False
        self._editing_verify = False
        self._pending_proposals = []
        chat.clear()
        self._render_history()
        meta = self.agent.session_meta
        chat.add_info(f"已切换到会话 {meta.id} · {meta.title or '(无标题)'}")
        self._report_session()

    def _render_history(self) -> None:
        """Render the agent's restored history into the chat flow.

        User/assistant text is rendered as-is; raw tool outputs are summarized
        (tool cards are not rebuilt for history).
        """
        chat = self.query_one("#chat", ChatWidget)
        skipped_tools = 0
        for msg in self.agent.messages[1:]:  # skip the system message
            if is_summary_message(msg):
                chat.add_info("（此前对话已压缩为摘要）")
            elif msg.role == "user" and msg.content:
                chat.add_user_message(msg.content, msg.attachments)
            elif msg.role == "assistant" and msg.content:
                chat.add_assistant_message(msg.content)
            elif msg.role == "tool":
                skipped_tools += 1
        if skipped_tools:
            chat.add_info(f"（已省略 {skipped_tools} 条历史工具输出）")

    def _export_session(self, arg: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        meta = self.agent.session_meta
        if not meta.title:
            meta.title = derive_title(self.agent.messages)
        if arg:
            out = Path(arg).expanduser()
        else:
            out = (
                Path.home() / ".limbo" / "exports" / f"{self.agent.session_id}.jsonl"
            )
        try:
            if out.suffix == ".md":
                export_markdown(meta, self.agent.messages, out)
            else:
                export_jsonl(
                    meta, self.agent.messages, out, trace_path=self.agent.trace.path
                )
        except OSError as e:
            chat.add_error(f"导出失败：{e}")
            return
        chat.add_info(f"已导出到 {out}")

    def action_toggle_tools(self) -> None:
        self.query_one("#chat", ChatWidget).toggle_tool_bodies()

    def action_handle_escape(self) -> None:
        """Screen-level ESC binding: fires only when the input widget is
        NOT focused (its priority binding consumes ESC first otherwise)."""
        self.handle_escape()

    def _btw(self, question: str) -> None:
        """Open the /btw side-question overlay.

        Works mid-turn (allow_when_busy): the query runs on a snapshot of
        the history and the answer lives only in the overlay — the steer
        queue and session history are never touched (see limbo.btw).
        """
        chat = self.query_one("#chat", ChatWidget)
        if not question:
            chat.add_info("用法：/btw <问题>")
            return
        self.app.push_screen(
            BtwScreen(
                question=question,
                llm_client=self.llm_client,
                history=list(self.agent.messages),
                trace=self.agent.trace,
            )
        )

    def _open_game2048(self) -> None:
        """Open the 2048 modal. The agent turn (if any) keeps running."""
        self.app.push_screen(Game2048Screen())

    def action_game2048(self) -> None:
        self._open_game2048()

    async def on_unmount(self) -> None:
        """Close owned resources on shutdown: the LLM client's HTTP pool and
        the agent's trace file handle."""
        self._integrations.release()
        self.agent.close()
        close = getattr(self.llm_client, "close", None)
        if close is not None:
            await close()

    def on_paste_markers_invalid(self, event: PasteMarkersInvalid) -> None:
        """Warn when a paste placeholder lost its content (undo / lookalike).

        The marker is submitted as literal text; surface the loss instead
        of letting the paste vanish silently.
        """
        ids = "、".join(f"#{i}" for i in event.paste_ids)
        chat = self.query_one("#chat", ChatWidget)
        chat.add_info(f"⚠ 粘贴 {ids} 内容已失效，将以字面文本发送")

    def on_user_submitted(self, event: UserSubmitted) -> None:
        text = event.message
        if self._editing_verify:
            # M2 edit mode: the submission is a verify command, not a chat
            # message. A slash command cancels edit mode and runs normally.
            self._editing_verify = False
            chat = self.query_one("#chat", ChatWidget)
            if text.startswith("/") and not event.force_text:
                self._report_state("idle")
                self._handle_command(text)
                return
            command = text.strip()
            if command:
                chat.add_user_message(f"验收命令：{command}")
                self._accept_verify(command)
            else:
                chat.add_info("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
                self._report_state("idle")
            return
        attachments = event.attachments
        if text.startswith("/") and not event.force_text:
            if self._handle_command(text):
                return
            # '/'-leading input that is not a command but looks like a path:
            # fall through as a normal message, auto-attaching existing files.
            attachments = [*attachments, *attachments_from_text(text)]
        chat = self.query_one("#chat", ChatWidget)
        if self._agent_busy:
            # Mid-turn submission (RFC LIM-20): queue for steer injection,
            # render optimistically — never start a second turn worker.
            item_id = self.agent.steer(text, attachments)
            chat.add_queued_message(item_id, text, attachments)
            self._update_queue_status()
            return
        chat.add_user_message(text, attachments)
        self.run_worker(self._handle_turn(text, attachments))

    # -- steer queue UI (LIM-20) ---------------------------------------------

    def _update_queue_status(self) -> None:
        self.query_one("#statusbar", StatusBar).set_queued(self.agent.queued_count)

    def cancel_queued(self, item_id: str) -> None:
        """Cancel one queued steer message (the card's ✕ affordance)."""
        chat = self.query_one("#chat", ChatWidget)
        if self.agent.cancel_steer(item_id):
            chat.mark_steer_cancelled(item_id)
        else:
            # Already drained past the injection boundary (or unknown id).
            chat.add_info("该消息已注入，无法撤回")
        self._update_queue_status()

    def handle_escape(self) -> None:
        """ESC routing chain (RFC LIM-53), in priority order.

        Entry points: the input widget's priority binding (input focused)
        and this screen's own ``escape`` binding (focus anywhere else).
        Modal screens handle ESC in their own bindings. The chain,
        highest priority first: close an open slash menu, cancel an
        in-flight verify subprocess (LIM-40), interrupt the running
        turn/compact worker (LIM-53), then fall back to cancelling the
        newest queued steer message (LIM-20). Note the deliberate
        behavior change: while a turn is running, ESC interrupts the
        turn — queued-steer cancel is the card's ✕ affordance during
        that window.
        """
        # Layer 1 lives here (not only in the input binding) so the narrow
        # path "menu open → click the chat (focus moves, menu stays open)
        # → ESC" still closes the menu first instead of skipping a layer.
        if self.slash_menu_open:
            self.slash_menu_close()
            return
        if self.pump.verifying:
            if self.pump.cancel_verify():
                self.query_one("#chat", ChatWidget).add_info("已取消验收命令")
            return
        if self._agent_busy:
            self.agent.interrupt()
            return
        item_id = self.agent.cancel_latest_steer()
        if item_id is None:
            return
        self.query_one("#chat", ChatWidget).mark_steer_cancelled(item_id)
        self._update_queue_status()

    def on_click(self, event) -> None:
        """Clicks on a queued card's ✕ cancel that steer message."""
        widget = event.widget
        if not isinstance(widget, Static) or not widget.has_class("queued-cancel"):
            return
        node = widget.parent
        while node is not None and not isinstance(node, QueuedMessage):
            node = node.parent
        if node is not None:
            self.cancel_queued(node.item_id)

    async def _handle_turn(
        self, user_input: str, attachments: list[Attachment] | None = None
    ) -> None:
        """Adapter: pump one user input, translating events into UI updates.

        All orchestration — single-flight, the follow-up steer drain, the
        goal loop — lives in the TurnPump (limbo.pump); this worker only
        owns what the user sees: status bar, focus, the goal badge, and
        the post-turn verify-proposal picker.
        """
        input_widget = self.query_one("#input", InputWidget)
        statusbar = self.query_one("#statusbar", StatusBar)
        # The input stays enabled during the turn (RFC LIM-20): submissions
        # go to the steer queue instead of being blocked. Single-flight is
        # owned by the pump (``running`` is set eagerly on call).
        statusbar.set_state("thinking…", "thinking")
        self._report_state("working")
        # pump.run() sets ``running`` eagerly at call time, so the badge
        # refresh below already sees the loop as executing (rainbow on).
        events = self.pump.run(user_input, attachments)
        self._refresh_goal_indicator()
        try:
            async for event in events:
                await self._process_agent_event(event)
        except Exception as e:  # noqa: BLE001
            # A pump crash is a bug (the agent loop converts LLM/tool
            # failures into error events). Surface it instead of silently
            # swallowing it the way the old finally-drain did.
            self.query_one("#chat", ChatWidget).add_error(f"内部错误：{e}")
        finally:
            self._update_queue_status()
            # Disabling the input mid-turn moves focus away; give it back so
            # the user can keep typing without clicking.
            input_widget.focus()
            statusbar.set_state("idle")
            self._report_state("idle")
            self._refresh_goal_indicator()
            # M2: a proposal round that ends without leftover steers pops
            # the verify-command confirmation picker.
            self._maybe_confirm_verify_proposal()

    async def _process_agent_event(self, event: PumpEvent) -> None:
        chat = self.query_one("#chat", ChatWidget)
        statusbar = self.query_one("#statusbar", StatusBar)

        if isinstance(event, TextDelta):
            await chat.append_assistant_text(event.text)
        elif isinstance(event, ThinkingDelta):
            await chat.append_thinking_text(event.text)
        elif isinstance(event, ErrorEvent):
            chat.add_error(event.message)
        elif isinstance(event, CompactionEvent):
            if event.compacted:
                chat.add_info(
                    f"已压缩上下文：约 {event.before_tokens} → "
                    f"{event.after_estimate} tokens"
                    + ("（手动）" if event.trigger == "manual" else "（自动）")
                )
            elif event.reason:
                chat.add_info(event.reason)
            if event.warning:
                chat.add_info(event.warning)
        elif isinstance(event, UsageUpdate):
            statusbar.set_tokens(event.total_tokens)
        elif isinstance(event, SteerEvent):
            chat.mark_steer_delivered(event.id)
            self._update_queue_status()
        elif isinstance(event, InterruptEvent):
            # RFC LIM-53: cards still running belong to calls that will
            # never execute (the stream was cut mid-tool-call) — mark them.
            chat.cancel_running_tool_cards()
            chat.add_info("⏹ 已打断")
        elif isinstance(event, GoalVerifyStarted):
            chat.add_info(
                f"🔍 第 {event.round}/{event.max_rounds} 轮验收：`{event.command}`"
            )
            statusbar.set_state("verifying…", "tool")
        elif isinstance(event, GoalVerifyResultEvent):
            vresult = event.result
            if vresult.cancelled:
                chat.add_info("验收已取消，闭环暂停（goal 保持 active）")
            elif vresult.timed_out:
                chat.add_error("验收命令执行超时，将带入下一轮处理")
            elif vresult.exit_code == 0:
                chat.add_info("✅ 验收命令退出码 0")
            else:
                chat.add_error(
                    f"❌ 第 {event.round} 轮验收未通过（退出码 "
                    f"{vresult.exit_code}），失败输出将原样注入下一轮"
                )
            statusbar.set_state("thinking…", "thinking")
            self._refresh_goal_indicator()
        elif isinstance(event, GoalPassed):
            chat.add_info(f"✅ 目标达成（共 {event.rounds} 轮验收）")
            self._refresh_goal_indicator()
        elif isinstance(event, GoalExhausted):
            chat.add_info(
                f"⚠️ 已达最大轮次（{event.rounds} 轮）：发送任意消息带新预算继续，"
                "/goal clear 退出"
            )
            self._refresh_goal_indicator()
        elif isinstance(event, GoalResumed):
            chat.add_info(f"🎯 恢复 goal 闭环，预算已重置（{event.max_rounds} 轮）")
            self._refresh_goal_indicator()
        elif isinstance(event, ToolCallRequest):
            chat.add_tool_card(event.id, event.name, event.arguments)
            statusbar.set_state(f"running {event.name}…", "tool")
        elif isinstance(event, ToolResultEvent):
            result = event.result
            card = chat.add_tool_card(event.id, event.name, event.arguments)
            if result.success:
                card.set_success(result.output or "")
            else:
                card.set_error(result.error or "Tool failed.")
            # Success or failure, the turn continues with another LLM call
            # that reacts to the result — the bar must stay busy. (Showing
            # idle here made mid-turn submissions look like the agent was
            # stuck: they steer-queued while the bar claimed idle.)
            statusbar.set_state("thinking…", "thinking")
