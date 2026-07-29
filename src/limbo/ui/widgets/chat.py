"""Chat message flow widget (pi-style single column).

Renders the conversation as a top-to-bottom stream: user messages, assistant
Markdown (streamed), inline tool-call cards, and error/info lines.
"""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
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
        self._current_thinking: Static | None = None
        # Scroll-follow state: when the user scrolls up, new content no longer
        # drags the view down; a "back to bottom" pill appears instead (P2-4).
        self._follow = True
        self._pending_count = 0

    def compose(self) -> ComposeResult:
        # Docked to the bottom edge of the scroll view; hidden until needed.
        pill = Static("", id="back-to-bottom", markup=False)
        pill.display = False
        yield pill

    # -- messages -----------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self._current_assistant = None
        self._current_thinking = None
        msg = Static(f"❯ {text}", classes="user-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    def add_info(self, text: str) -> None:
        msg = Static(text, classes="info-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    def add_art(self, text: str | Text) -> None:
        """Add preformatted ASCII art (e.g. the startup banner) verbatim.

        Accepts a Rich ``Text`` for per-character colors; plain strings are
        rendered without markup.
        """
        msg = Static(text, classes="ascii-art", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    def add_error(self, text: str) -> None:
        self._current_assistant = None
        self._current_thinking = None
        # Structured error block (RFC §6.3): ✗ marker + indented detail lines.
        lines = text.splitlines() or [""]
        body = "✗ " + lines[0]
        if len(lines) > 1:
            body += "\n" + "\n".join("  " + line for line in lines[1:])
        msg = Static(body, classes="error-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)

    async def append_thinking_text(self, text: str) -> None:
        """Append a streamed reasoning chunk to the current thinking block.

        Rendered as muted plain text (not Markdown) to keep thinking visually
        secondary to the assistant's reply.
        """
        if self._current_thinking is None:
            # Thinking after assistant text starts a new block.
            self._current_assistant = None
            block = Static("", classes="thinking-message", markup=False)
            self.messages.append(block)
            self._current_thinking = block
            await self.mount(block)
        self._current_thinking.update(str(self._current_thinking.content) + text)
        self._auto_scroll()

    async def append_assistant_text(self, text: str) -> None:
        """Append a streamed chunk to the current assistant Markdown block."""
        # Assistant text after thinking starts a new block.
        self._current_thinking = None
        if self._current_assistant is None:
            # markdown=None: on_mount applies the initial markdown and would
            # wipe any chunks appended before mounting finished, so the mount
            # must be awaited before the first append.
            md = Markdown(classes="assistant-message")
            self.messages.append(md)
            self._current_assistant = md
            await self.mount(md)
        # Markdown.append() mutates its source synchronously but defers the
        # re-render via AwaitComplete. Not awaiting it lets fast chunk bursts
        # queue multiple stale renders that re-mount existing blocks
        # (visually duplicated content), so each append is awaited.
        await self._current_assistant.append(text)
        self._auto_scroll()

    def add_assistant_message(self, text: str) -> None:
        """Add a complete (non-streamed) assistant Markdown block."""
        self._current_assistant = None
        md = Markdown(text, classes="assistant-message")
        self.messages.append(md)
        self._mount_and_scroll(md)

    def clear(self) -> None:
        """Remove all rendered messages and tool cards (keeps the floater)."""
        for child in list(self.children):
            if child.id != "back-to-bottom":
                child.remove()
        self.messages.clear()
        self.tool_cards.clear()
        self._current_assistant = None
        self._current_thinking = None
        self._pending_count = 0
        self._follow = True
        self._update_floater()

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
        self._current_thinking = None
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
        self._auto_scroll()

    # -- scroll follow (P2-4) --------------------------------------------------

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Follow the tail only while the user stays at the bottom."""
        super().watch_scroll_y(old_value, new_value)
        self._follow = self.max_scroll_y - new_value <= 2
        if self._follow:
            self._pending_count = 0
        self._update_floater()

    def _auto_scroll(self) -> None:
        if self._follow:
            self.scroll_end(animate=False)
        else:
            self._pending_count += 1
        self._update_floater()

    def _update_floater(self) -> None:
        try:
            pill = self.query_one("#back-to-bottom", Static)
        except NoMatches:
            return  # Not composed yet.
        if self._pending_count and not self._follow:
            pill.update(f"↓ {self._pending_count} 条新消息 · 点击回到底部")
            pill.display = True
        else:
            pill.display = False

    def on_click(self, event) -> None:
        if getattr(event.widget, "id", None) == "back-to-bottom":
            self._pending_count = 0
            self.scroll_end(animate=True)
