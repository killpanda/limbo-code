"""Main screen: pi-style single-column chat layout."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

from limbo.agent import (
    Agent,
    AgentEvent,
    ErrorEvent,
    TextDelta,
    ToolCallRequest,
    ToolResultEvent,
)
from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.confirm import ConfirmDialog
from limbo.ui.widgets.input import InputWidget, UserSubmitted
from limbo.ui.widgets.status_bar import StatusBar

CONFIRMATION_TIMEOUT = 300.0


class MainScreen(Screen[None]):
    """Single-column chat screen: status bar / chat flow / input / hint."""

    BINDINGS = [
        Binding("ctrl+o", "toggle_tools", "展开/收起工具输出"),
    ]

    def __init__(
        self,
        workdir: Path,
        config: Config | None = None,
        llm_client: LLMClient | None = None,
        session_dir: Path | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workdir = workdir
        self.config = config or Config()
        self.llm_client = llm_client or OpenAICompatibleClient(self.config)
        self.agent = Agent(
            config=self.config,
            llm_client=self.llm_client,
            workdir=workdir,
            session_dir=session_dir,
        )
        self._confirmation_event = asyncio.Event()
        self._confirmation_result: bool | None = None

    def compose(self) -> ComposeResult:
        yield StatusBar(
            model=self.config.llm.model,
            workdir=str(self.workdir),
            id="statusbar",
        )
        yield ChatWidget(id="chat")
        yield InputWidget(id="input")
        yield Static(
            "Enter 发送 · Shift+Enter 换行 · ctrl+o 展开/收起工具输出",
            id="hint",
            markup=False,
        )

    def on_mount(self) -> None:
        chat = self.query_one("#chat", ChatWidget)
        chat.add_info(f"Limbo ready · {self.config.llm.model} · {self.workdir}")
        self.query_one("#input", InputWidget).focus()

    def action_toggle_tools(self) -> None:
        self.query_one("#chat", ChatWidget).toggle_tool_bodies()

    def handle_confirmation(self) -> None:
        """Handle an approval from the confirmation dialog."""
        self._confirmation_result = True
        self._confirmation_event.set()

    def handle_rejection(self) -> None:
        """Handle a rejection from the confirmation dialog."""
        self._confirmation_result = False
        self._confirmation_event.set()

    async def on_unmount(self) -> None:
        """Close the LLM client on shutdown to release its HTTP resources."""
        if isinstance(self.llm_client, OpenAICompatibleClient):
            await self.llm_client.close()

    def on_user_submitted(self, event: UserSubmitted) -> None:
        chat = self.query_one("#chat", ChatWidget)
        chat.add_user_message(event.message)
        self.run_worker(self._handle_turn(event.message))

    async def _handle_turn(self, user_input: str) -> None:
        input_widget = self.query_one("#input", InputWidget)
        statusbar = self.query_one("#statusbar", StatusBar)
        input_widget.disabled = True
        statusbar.set_state("thinking…", "thinking")
        try:
            stream: AsyncIterator[AgentEvent] = self.agent.run(user_input)
            # The agent may pause for confirmation mid-turn. When a tool is
            # confirmed, `confirmation_applied` becomes True and the stream is
            # replaced with the continuation so remaining tools/responses are
            # processed before returning to the user.
            while True:
                async for event in stream:
                    await self._process_agent_event(event)

                if self.agent.confirmation_applied:
                    stream = self.agent.continue_after_confirmation()
                    continue
                break
        finally:
            input_widget.disabled = False
            statusbar.set_state("idle")

    async def _process_agent_event(self, event: AgentEvent) -> None:
        chat = self.query_one("#chat", ChatWidget)
        statusbar = self.query_one("#statusbar", StatusBar)
        input_widget = self.query_one("#input", InputWidget)

        if isinstance(event, TextDelta):
            await chat.append_assistant_text(event.text)
        elif isinstance(event, ErrorEvent):
            chat.add_error(event.message)
        elif isinstance(event, ToolCallRequest):
            chat.add_tool_card(event.id, event.name, event.arguments)
            statusbar.set_state(f"running {event.name}…", "tool")
        elif isinstance(event, ToolResultEvent):
            result = event.result
            card = chat.add_tool_card(event.id, event.name, event.arguments)

            if not result.requires_confirmation:
                if result.success:
                    card.set_success(result.output or "")
                    statusbar.set_state("thinking…", "thinking")
                else:
                    card.set_error(result.error or "Tool failed.")
                    statusbar.set_state("idle")
                return

            # --- confirmation flow ---
            card.set_pending(result.output or "")
            input_widget.disabled = True
            self._confirmation_event.clear()
            self._confirmation_result = None
            warning = None
            if event.name == "bash":
                warning = (
                    "bash 安全过滤仅是启发式的，可能被子 shell 或命令替换绕过，"
                    "请仔细审查后再确认。"
                )
            dialog = ConfirmDialog(
                title=f"Apply {event.name}?",
                body=result.output or "",
                warning=warning,
            )
            self.app.push_screen(dialog)
            try:
                await asyncio.wait_for(
                    self._confirmation_event.wait(),
                    timeout=CONFIRMATION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                chat.add_error(f"[{event.name} 确认超时，已自动拒绝]")
                card.set_rejected()
                dialog.dismiss()
                input_widget.disabled = False
                self.agent.reject_pending_tool()
                return

            if self._confirmation_result:
                apply_result = await self.agent.apply_tool(
                    event.name, event.arguments
                )
                if apply_result.success:
                    card.set_applied(apply_result.output or "")
                else:
                    card.set_error(apply_result.error or "Tool failed.")
            else:
                card.set_rejected()
                self.agent.reject_pending_tool()
