"""Main screen with three-column layout."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from limbo.agent import Agent
from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.llm.openai_client import OpenAICompatibleClient
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.confirm import ConfirmDialog, Confirmed, Rejected
from limbo.ui.widgets.file_preview import FilePreviewWidget
from limbo.ui.widgets.input import InputWidget, UserSubmitted
from limbo.ui.widgets.sidebar import SidebarWidget


class MainScreen(Screen[None]):
    """Main three-column layout."""

    def __init__(
        self,
        workdir: Path,
        config: Config | None = None,
        llm_client: LLMClient | None = None,
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

    def on_confirmed(self, _event: Confirmed) -> None:
        self._confirmation_result = True
        self._confirmation_event.set()

    def on_rejected(self, _event: Rejected) -> None:
        self._confirmation_result = False
        self._confirmation_event.set()

    def on_user_submitted(self, event: UserSubmitted) -> None:
        chat = self.query_one("#chat", ChatWidget)
        chat.add_user_message(event.message)
        self.run_worker(self._handle_turn(event.message))

    async def _handle_turn(self, user_input: str) -> None:
        chat = self.query_one("#chat", ChatWidget)
        sidebar = self.query_one("#sidebar", SidebarWidget)
        preview = self.query_one("#preview", FilePreviewWidget)

        for event in self.agent.run(user_input):
            if hasattr(event, "text"):
                chat.append_assistant_text(event.text)
            elif hasattr(event, "name") and hasattr(event, "result"):
                result = event.result
                sidebar.set_status(f"Tool: {event.name}")
                if result.output:
                    preview.show(f"{event.name} result", result.output)

                if result.requires_confirmation:
                    self._confirmation_event.clear()
                    self._confirmation_result = None
                    self.app.push_screen(
                        ConfirmDialog(
                            title=f"Apply {event.name}?",
                            body=result.output or "",
                        )
                    )
                    await self._confirmation_event.wait()
                    if self._confirmation_result:
                        apply_result = self.agent.apply_tool(event.name, event.arguments)
                        preview.show(
                            f"{event.name} applied",
                            apply_result.output or "",
                        )
                    else:
                        chat.append_assistant_text(f"\n[{event.name} was rejected by user.]")
