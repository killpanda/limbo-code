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
from limbo.goal_driver import (
    DriverEvent,
    GoalDriver,
    GoalExhausted,
    GoalPassed,
    GoalResumed,
    GoalVerifyResultEvent,
    GoalVerifyStarted,
)
from limbo.llm.client import LLMClient
from limbo.llm.factory import create_llm_client
from limbo.model_switch import (
    prepare_model_switch,
    reload_llm_config,
    swap_llm_client,
)
from limbo.models import Attachment
from limbo.sessions import derive_title, export_jsonl, export_markdown, list_sessions
from limbo.skills import Skill, discover_skills
from limbo.tools.bash import is_dangerous
from limbo.ui.banner import startup_art_text
from limbo.ui.commands import SlashCommand, SlashCommandRegistry
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
from limbo.user_paths import extract_grantable_paths


class MainScreen(Screen[None]):
    """Single-column chat screen: status bar / chat flow / input / hint."""

    BINDINGS = [
        Binding("ctrl+o", "toggle_tools", "展开/收起工具输出"),
        Binding("ctrl+g", "game2048", "2048 小游戏"),
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
        self.driver = self._new_driver(self.agent)
        self._slash_menu_open = False
        # M2: proposal-round bookkeeping. _awaiting_verify_proposal is set
        # when /goal starts a proposal round; the confirmation picker pops
        # at the next quiet turn boundary. _editing_verify marks that the
        # next submission is a verify command being edited (not a message).
        self._awaiting_verify_proposal = False
        self._editing_verify = False
        self._pending_proposals: list[str] = []
        # Busy while a /compact worker owns the agent history; turn-level
        # busy is owned by the goal driver (see _agent_busy).
        self._compact_busy = False
        self._commands = SlashCommandRegistry()
        self._register_builtin_commands()

    @property
    def _agent_busy(self) -> bool:
        """Busy while a driver run OR a /compact worker owns the history.

        Turn-level single-flight lives in ``GoalDriver.running`` (eagerly
        set when ``driver.run()`` is called); /compact sets its own flag
        because it bypasses the driver.
        """
        return self._compact_busy or self.driver.running

    def _new_driver(self, agent: Agent) -> GoalDriver:
        return GoalDriver(
            agent,
            self.workdir,
            max_rounds=self.config.goal.max_rounds,
            verify_timeout_ms=self.config.goal.verify_timeout_ms,
            dangerous_patterns=self.config.safety.dangerous_commands,
        )

    def _new_agent(self, resume: Path | None = None) -> Agent:
        return Agent(
            config=self.config,
            llm_client=self.llm_client,
            workdir=self.workdir,
            session_dir=self.session_dir,
            resume=resume,
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
            if matches:
                menu = self.query_one("#slash-menu", SlashCommandMenu)
                menu.show_commands(matches)
                self._slash_menu_open = True
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
            state = self.driver.status()
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
            if self.driver.status() is None:
                chat.add_info("当前没有 goal")
                return
            self.driver.clear()
            self._awaiting_verify_proposal = False
            self._editing_verify = False
            suffix = "（当前轮跑完后停止续轮）" if self.driver.running else ""
            chat.add_info(f"已退出 goal 模式{suffix}")
            self._refresh_goal_indicator()
            return
        # GoalSet: starting a new goal is a history-shaping action, so it
        # follows the same busy rule as other rewriting commands (LIM-20).
        if self._agent_busy:
            chat.add_info("当前任务进行中，请等待完成后再设定 /goal")
            return
        state = self.driver.set_goal(command.text)
        chat.add_user_message(f"/goal {command.text}")
        self._refresh_goal_indicator()
        # M2: the first round is a proposal round — the model explores the
        # repo and proposes the acceptance command(s); the confirmation
        # picker pops when the turn ends (see _maybe_confirm_verify_proposal).
        self._awaiting_verify_proposal = True
        self.run_worker(self._handle_turn(build_proposer_prompt(state)))

    def _refresh_goal_indicator(self) -> None:
        """Sync the status-bar badge with driver state (rainbow = running)."""
        statusbar = self.query_one("#statusbar", StatusBar)
        state = self.driver.status()
        if state is not None and state.status == STATUS_ACTIVE:
            statusbar.set_goal(
                (state.rounds_completed, state.max_rounds),
                running=self.driver.running,
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
        state = self.driver.status()
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
        self.app.push_screen(
            VerifyPicker(self._pending_proposals), self._on_verify_picked
        )

    def _on_verify_picked(self, choice: str | None) -> None:
        chat = self.query_one("#chat", ChatWidget)
        self._awaiting_verify_proposal = False
        state = self.driver.status()
        if state is None or state.status != STATUS_ACTIVE:
            return
        if choice is None:
            chat.add_info("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            self._refresh_goal_indicator()
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
            return
        self._accept_verify(choice)

    def _accept_verify(self, command: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        if is_dangerous(command, self.config.safety.dangerous_commands):
            chat.add_error(f"验收命令命中危险命令过滤，已拒绝：{command}")
            chat.add_info("goal 保持单轮模式；/goal clear 退出")
            return
        state = self.driver.set_verify(command)
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

    def _handle_command(self, text: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        name, _, arg = text.partition(" ")
        arg = arg.strip()

        command = self._commands.get(name.lower())
        if command is not None and command.handler is not None:
            # Busy guard (RFC LIM-20): history-rewriting commands are
            # rejected mid-turn. Centralized here so both entry points
            # (on_user_submitted and slash_menu_complete) are covered.
            if self._agent_busy and not command.allow_when_busy:
                self._reject_busy_command(command)
                return
            command.handler(arg)
            return
        skill = self._find_skill(name.removeprefix("/"))
        if skill is not None:
            self._invoke_skill(skill, arg)
        else:
            chat.add_info(f"未知命令 {name}，{self._commands.help_text()}")

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
        # Same busy discipline as _handle_turn: the summary call takes
        # seconds, and a concurrent turn or second /compact would race
        # compact()'s wholesale history rewrite and silently drop messages.
        input_widget.disabled = True
        self._compact_busy = True
        statusbar.set_state("compacting…", "thinking")
        try:
            async for event in self.agent.compact(trigger="manual"):
                await self._process_agent_event(event)
        finally:
            self._compact_busy = False
            input_widget.disabled = False
            input_widget.focus()
            statusbar.set_state("idle")

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
        chat.clear()
        chat.add_info(f"已开始新会话 {self.agent.session_id}")

    def _find_skill(self, name: str) -> Skill | None:
        if self._commands.get(f"/{name}") is not None:
            return None
        for skill in discover_skills(self.workdir):
            if skill.name == name:
                return skill
        return None

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
        self.driver = self._new_driver(self.agent)
        self._awaiting_verify_proposal = False
        self._editing_verify = False
        self._pending_proposals = []
        chat.clear()
        self._render_history()
        meta = self.agent.session_meta
        chat.add_info(f"已切换到会话 {meta.id} · {meta.title or '(无标题)'}")

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
            if text.startswith("/"):
                self._handle_command(text)
                return
            command = text.strip()
            if command:
                chat.add_user_message(f"验收命令：{command}")
                self._accept_verify(command)
            else:
                chat.add_info("已跳过自动验收（单轮模式）；/goal clear 可退出 goal")
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        chat = self.query_one("#chat", ChatWidget)
        if self._agent_busy:
            # Mid-turn submission (RFC LIM-20): queue for steer injection,
            # render optimistically — never start a second turn worker.
            # Queued steer messages are genuine user input too: grant the
            # paths they reference (LIM-19), since the model will see them
            # at injection time just like a normal submission.
            self._grant_user_paths(text, event.attachments)
            item_id = self.agent.steer(text, event.attachments)
            chat.add_queued_message(item_id, text, event.attachments)
            self._update_queue_status()
            return
        chat.add_user_message(text, event.attachments)
        self._grant_user_paths(text, event.attachments)
        self.run_worker(self._handle_turn(text, event.attachments))

    def _grant_user_paths(self, text: str, attachments: list[Attachment]) -> None:
        """Implicit grants: existing paths in a human-submitted message (and
        its attachments) widen the file-tool fence for this session.

        Runs only on the real user-submit event — never on model text — so
        the grant source is always genuine user input. Grants are visible
        in the chat and traced; they persist in the session meta.
        """
        if not self.config.safety.auto_grant_user_paths:
            return
        candidates = extract_grantable_paths(text)
        candidates.extend(
            path
            for attachment in attachments
            if (path := Path(attachment.path)).exists()
        )
        new_roots = self.agent.registry.add_allowed_roots(candidates)
        chat = self.query_one("#chat", ChatWidget)
        for root in new_roots:
            chat.add_info(f"↳ 已允许访问：{root}（本会话有效）")
            self.agent.trace.log("path_grant", root=str(root), source="user_message")

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

    def cancel_latest_queued(self) -> None:
        """Esc with no menu open: cancel verify first, then the newest
        queued steer message (LIM-40: during the verify window Esc routes
        to the verify subprocess, not the steer queue)."""
        if self.driver.verifying:
            if self.driver.cancel_verify():
                self.query_one("#chat", ChatWidget).add_info("已取消验收命令")
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
        input_widget = self.query_one("#input", InputWidget)
        statusbar = self.query_one("#statusbar", StatusBar)
        # The input stays enabled during the turn (RFC LIM-20): submissions
        # go to the steer queue instead of being blocked. Single-flight is
        # owned by the goal driver (``running`` is set eagerly on call).
        statusbar.set_state("thinking…", "thinking")
        # driver.run() sets ``running`` eagerly at call time, so the badge
        # refresh below already sees the loop as executing (rainbow on).
        events = self.driver.run(user_input, attachments)
        self._refresh_goal_indicator()
        try:
            async for event in events:
                await self._process_agent_event(event)
        finally:
            chat = self.query_one("#chat", ChatWidget)
            # Automatic follow-up turn (RFC LIM-20): drain + hand off BEFORE
            # releasing the busy flag, so no user submission can slip into
            # the gap and start a second concurrent turn worker. Leftovers
            # happen on error / max-iterations exits (the loop's own
            # consumption points never leave the queue non-empty otherwise).
            pending = self.agent.drain_steer()
            if pending:
                for item in pending:
                    chat.mark_steer_delivered(item.id)
                joined = "\n\n".join(item.text for item in pending)
                followup_attachments = [
                    a for item in pending for a in item.attachments
                ]
                self._update_queue_status()
                self.run_worker(self._handle_turn(joined, followup_attachments))
                return
            self._update_queue_status()
            # Disabling the input mid-turn moves focus away; give it back so
            # the user can keep typing without clicking.
            input_widget.focus()
            statusbar.set_state("idle")
            self._refresh_goal_indicator()
            # M2: a proposal round that ends without leftover steers pops
            # the verify-command confirmation picker.
            self._maybe_confirm_verify_proposal()

    async def _process_agent_event(self, event: DriverEvent) -> None:
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
        elif isinstance(event, GoalVerifyStarted):
            chat.add_info(
                f"🔍 第 {event.round}/{event.max_rounds} 轮验收：`{event.command}`"
            )
            statusbar.set_state("verifying…", "tool")
        elif isinstance(event, GoalVerifyResultEvent):
            vresult = event.result
            if vresult.refused:
                chat.add_error("验收命令命中危险命令过滤，闭环暂停（goal 保持 active）")
            elif vresult.cancelled:
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
