"""Main screen with three-column layout."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from limbo.agent import (
    Agent,
    AgentEvent,
    ErrorEvent,
    TextDelta,
    ToolResultEvent,
)
from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.confirm import ConfirmDialog
from limbo.ui.widgets.file_preview import FilePreviewWidget
from limbo.ui.widgets.input import InputWidget, UserSubmitted
from limbo.ui.widgets.sidebar import SidebarWidget

CONFIRMATION_TIMEOUT = 300.0


class MainScreen(Screen[None]):
    """Main three-column layout."""

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

    def compose(self):
        with Horizontal():
            with Vertical(id="sidebar-container"):
                yield SidebarWidget(id="sidebar")
            with Vertical(id="chat-container"):
                yield ChatWidget(id="chat")
                yield InputWidget(id="input")
            with Vertical(id="preview-container"):
                yield FilePreviewWidget(id="preview")

    def _normalized_recent_path(self, path: str) -> str | None:
        """Return a workdir-relative path for the recent-files sidebar.

        Returns ``None`` when the path cannot be resolved inside the workdir.
        """
        try:
            resolved = (self.workdir / path).resolve()
            rel = resolved.relative_to(self.workdir)
            return str(rel)
        except (ValueError, OSError):
            return None

    def handle_confirmation(self) -> None:
        """Handle an approval from the confirmation dialog."""
        self._confirmation_result = True
        self._confirmation_event.set()

    def handle_rejection(self) -> None:
        """Handle a rejection from the confirmation dialog."""
        self._confirmation_result = False
        self._confirmation_event.set()

    def on_user_submitted(self, event: UserSubmitted) -> None:
        chat = self.query_one("#chat", ChatWidget)
        chat.add_user_message(event.message)
        self.run_worker(self._handle_turn(event.message))

    async def _handle_turn(self, user_input: str) -> None:
        input_widget = self.query_one("#input", InputWidget)
        input_widget.disabled = True
        try:
            stream: AsyncIterator[AgentEvent] = self.agent.run(user_input)
            while True:
                async for event in stream:
                    await self._process_agent_event(event)

                if self.agent.confirmation_applied:
                    stream = self.agent.continue_after_confirmation()
                    continue
                break
        finally:
            input_widget.disabled = False

    async def _process_agent_event(self, event: AgentEvent) -> None:
        chat = self.query_one("#chat", ChatWidget)
        sidebar = self.query_one("#sidebar", SidebarWidget)
        preview = self.query_one("#preview", FilePreviewWidget)
        input_widget = self.query_one("#input", InputWidget)

        if isinstance(event, TextDelta):
            chat.append_assistant_text(event.text)
        elif isinstance(event, ErrorEvent):
            chat.append_assistant_text(event.message)
        elif isinstance(event, ToolResultEvent):
            result = event.result
            sidebar.set_status(f"Tool: {event.name}")
            if not result.success:
                preview.show(
                    f"{event.name} error", result.error or "Tool failed."
                )
                chat.append_assistant_text(
                    f"\n[{event.name} failed: {result.error or 'unknown error'}]"
                )
            elif result.output:
                preview.show(f"{event.name} result", result.output)

            if (
                event.name == "read"
                and result.success
                and "path" in event.arguments
            ):
                recent = self._normalized_recent_path(event.arguments["path"])
                if recent is not None:
                    sidebar.add_recent_file(recent)

            if result.requires_confirmation:
                input_widget.disabled = True
                self._confirmation_event.clear()
                self._confirmation_result = None
                body = result.output or ""
                if event.name == "bash":
                    body += (
                        "\n\nWarning: the bash safety filter is heuristic and can"
                        " be bypassed by subshells or command substitution."
                        " Review carefully before confirming."
                    )
                dialog = ConfirmDialog(
                    title=f"Apply {event.name}?",
                    body=body,
                )
                self.app.push_screen(dialog)
                try:
                    await asyncio.wait_for(
                        self._confirmation_event.wait(),
                        timeout=CONFIRMATION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    chat.append_assistant_text(
                        f"\n[{event.name} confirmation timed out; action rejected.]"
                    )
                    dialog.dismiss()
                    input_widget.disabled = False
                    self.agent.reject_pending_tool()
                    return

                if self._confirmation_result:
                    apply_result = await self.agent.apply_tool(
                        event.name, event.arguments
                    )
                    if not apply_result.success:
                        preview.show(
                            f"{event.name} error",
                            apply_result.error or "Tool failed.",
                        )
                        chat.append_assistant_text(
                            f"\n[{event.name} failed: {apply_result.error or 'unknown error'}]"
                        )
                    else:
                        preview.show(
                            f"{event.name} applied",
                            apply_result.output or "",
                        )
                    path = event.arguments.get("path")
                    if path and apply_result.success:
                        recent = self._normalized_recent_path(path)
                        if recent is not None:
                            sidebar.add_recent_file(recent)
                else:
                    chat.append_assistant_text(
                        f"\n[{event.name} was rejected by user.]"
                    )
                    self.agent.reject_pending_tool()
