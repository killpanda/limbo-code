"""Chat message flow widget (pi-style single column).

Renders the conversation as a top-to-bottom stream: user messages, assistant
Markdown (streamed), inline tool-call cards, and error/info lines.
"""

from __future__ import annotations

from typing import Any

from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static

from limbo.ui.widgets.tool_card import ToolCard


class ChatWidget(VerticalScroll):
    """Displays the conversation as a single scrolling flow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Text-like messages (user/assistant/error/info) in display order.
        self.messages: list[Static | Markdown] = []
        # Tool cards keyed by tool-call id; ToolCallRequest events may arrive
        # twice for the same call (once streamed, once before execution).
        self.tool_cards: dict[str, ToolCard] = {}
        self._current_assistant: Markdown | None = None

    # -- messages -----------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self._current_assistant = None
        msg = Static(f"❯ {text}", classes="user-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    def add_info(self, text: str) -> None:
        msg = Static(text, classes="info-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    def add_error(self, text: str) -> None:
        self._current_assistant = None
        msg = Static(text, classes="error-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    async def append_assistant_text(self, text: str) -> None:
        """Append a streamed chunk to the current assistant Markdown block."""
        if self._current_assistant is None:
            # markdown=None: on_mount applies the initial markdown and would
            # wipe any chunks appended before mounting finished, so the mount
            # must be awaited before the first append.
            md = Markdown(classes="assistant-message")
            self.messages.append(md)
            self._current_assistant = md
            await self.mount(md)
        self._current_assistant.append(text)
        self.scroll_end(animate=False)

    # -- tool cards -----------------------------------------------------------

    def add_tool_card(
        self, tool_id: str, name: str, arguments: dict[str, Any]
    ) -> ToolCard:
        """Get or create the card for a tool call (idempotent by tool-call id)."""
        existing = self.tool_cards.get(tool_id)
        if existing is not None:
            return existing
        # Text after a tool card must start a new assistant block.
        self._current_assistant = None
        card = ToolCard(tool_id, name, arguments)
        self.tool_cards[tool_id] = card
        self._mount_and_scroll(card)
        return card

    def toggle_tool_bodies(self) -> None:
        """Expand/collapse all tool cards that have output."""
        for card in self.tool_cards.values():
            card.toggle()

    # -- helpers --------------------------------------------------------------

    def transcript_text(self) -> str:
        """Combined plain text of all text-like messages (for tests/debug)."""
        parts: list[str] = []
        for msg in self.messages:
            if isinstance(msg, Markdown):
                parts.append(msg.source)
            else:
                parts.append(str(msg.content))
        return "\n".join(parts)

    def _mount_and_scroll(self, widget: Any) -> None:
        self.mount(widget)
        self.scroll_end(animate=False)
