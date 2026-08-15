"""Chat message flow widget (pi-style single column).

Renders the conversation as a top-to-bottom stream: user messages, assistant
Markdown (streamed), inline tool-call cards, thinking blocks, and error/info
lines.

Performance model (long-session UI lag):
- Streaming output (assistant Markdown + thinking) is batched into ~12 fps
  time slices instead of re-rendering on every chunk: each tick performs at
  most one Markdown append, one thinking refresh, and one scroll.
- Thinking is collapsed by default to a one-line summary; the full text is
  only rendered while expanded (a click away).
- Old messages are pruned from the DOM beyond a window (``_MAX_DOM_MESSAGES``),
  keeping the widget tree bounded; the full history stays in ``self.messages``
  and is paged back in via the "load more" pill at the top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Markdown, Static

from limbo.models import Attachment
from limbo.ui.widgets.tool_card import ToolCard

# How many message widgets stay mounted in the DOM. Older ones are pruned
# (kept in ``self.messages``) and paged back via the "load more" pill.
_MAX_DOM_MESSAGES = 150
# How many pruned messages each "load more" click restores.
_PAGE_SIZE = 50
# Streaming flush rate: chunks accumulate and are rendered together at this
# cadence, capping full-document re-renders per second.
_FLUSH_INTERVAL = 1 / 12
# Summary line cap for a collapsed thinking block.
_THINKING_SUMMARY_CHARS = 60


class ThinkingBlock(Vertical):
    """Collapsible reasoning block.

    Collapsed (default) renders a one-line summary that updates cheaply
    while streaming; expanded renders the full text, materialized once per
    flush instead of per chunk. Click to toggle.
    """

    def __init__(self) -> None:
        super().__init__(classes="thinking-message")
        self._text = ""
        self._collapsed = True
        self._summary = Static("", classes="thinking-summary", markup=False)
        self._body = Static("", classes="thinking-body", markup=False)

    def compose(self) -> ComposeResult:
        yield self._summary
        yield self._body

    def on_mount(self) -> None:
        # Collapsed by default; if the block was pruned while expanded and
        # re-mounted via load-more, restore the expanded state.
        self._body.display = not self._collapsed
        self._summary.update(self._summary_text())
        if not self._collapsed:
            self._body.update(self._text)

    def append(self, text: str) -> None:
        self._text += text
        # Summary refresh is throttled to the flush cadence (via
        # ChatWidget._thinking_dirty): rebuilding the collapsed summary on
        # every chunk is O(full text) per chunk, and thinking is the
        # longest/highest-frequency stream — exactly what batching exists
        # to avoid. First-chunk feedback is at most one slice late.

    def refresh_display(self) -> None:
        """Re-render at most once per flush slice (see ChatWidget._tick_flush)."""
        if self._collapsed:
            self._summary.update(self._summary_text())
        else:
            self._body.update(self._text)

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._body.display = not self._collapsed
        self._summary.update(self._summary_text())
        if not self._collapsed:
            self._body.update(self._text)

    def on_click(self) -> None:
        self.toggle()

    @property
    def content(self) -> str:
        """Full reasoning text (transcript/debug)."""
        return self._text

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def _summary_text(self) -> str:
        n = len(self._text)
        # Truncate by display width (CJK chars are 2 cells), not char count.
        flat = " ".join(self._text.split())
        head = ""
        width = 0
        truncated = False
        for ch in flat:
            w = cell_len(ch)
            if width + w > _THINKING_SUMMARY_CHARS:
                truncated = True
                break
            width += w
            head += ch
        if self._collapsed:
            if not head:
                return f"🧠 thinking…（{n} 字）"
            ellipsis = "…" if truncated else ""
            return f"🧠 thinking… {head}{ellipsis}（{n} 字）"
        return f"🧠 thinking（点击折叠 · {n} 字）"


class QueuedMessage(Vertical):
    """A user message waiting in the agent's steer queue (RFC LIM-20).

    States: queued (⏳ status line with a ✕ cancel affordance) → delivered
    (status line removed, looks like a normal user message) or cancelled
    (dimmed, 「已取消」, kept in the flow as a record; never reaches the
    session history).

    ``compose`` derives everything from instance state so the widget can be
    pruned from the DOM and re-mounted (old-message paging) without losing
    its current state.
    """

    def __init__(
        self,
        item_id: str,
        text: str,
        attachments: list[Attachment] | None = None,
    ) -> None:
        super().__init__(classes="queued-message")
        self.item_id = item_id
        self._text = text
        self._attachments = attachments or []
        self.state = "queued"  # queued | delivered | cancelled

    def compose(self) -> ComposeResult:
        if self.state == "cancelled":
            self.add_class("cancelled")
        yield Static(f"❯ {self._text}", classes="user-message", markup=False)
        if self._attachments:
            chips = []
            for attachment in self._attachments:
                label = "图片" if attachment.kind == "image" else "文件"
                chips.append(f"{label}: {attachment.name}")
            yield Static(
                "📎 " + " · ".join(chips), classes="attachment-chips", markup=False
            )
        if self.state != "delivered":
            yield Horizontal(classes="queued-status")

    def on_mount(self) -> None:
        # Defensive: a widget can be pruned from the DOM before its first
        # compose completes (see ChatWidget._prune), in which case Textual
        # drops the composed children but still dispatches Mount — the
        # .queued-status query would then raise NoMatches and crash the app.
        try:
            if self.state == "queued":
                status = self.query_one(".queued-status", Horizontal)
                status.mount(Static("⏳ 排队中", classes="queued-hint", markup=False))
                status.mount(Static("✕ 取消", classes="queued-cancel", markup=False))
            elif self.state == "cancelled":
                status = self.query_one(".queued-status", Horizontal)
                status.mount(
                    Static("已取消", classes="queued-hint", markup=False)
                )
        except NoMatches:
            pass  # Pruned mid-mount; the widget is gone from the DOM anyway.

    def transcript(self) -> str:
        suffix = {"queued": "（排队中）", "cancelled": "（已取消）"}.get(self.state, "")
        text = f"❯ {self._text}{suffix}"
        if self._attachments:
            chips = []
            for attachment in self._attachments:
                label = "图片" if attachment.kind == "image" else "文件"
                chips.append(f"{label}: {attachment.name}")
            text += "\n📎 " + " · ".join(chips)
        return text

    def mark_delivered(self) -> None:
        """Flip to a normal user message (the agent injected it)."""
        if self.state != "queued":
            return
        self.state = "delivered"
        try:
            self.query_one(".queued-status", Horizontal).remove()
        except NoMatches:
            pass  # Not composed / already pruned.

    def mark_cancelled(self) -> None:
        """Flip to the cancelled record state (dimmed, no longer cancellable)."""
        if self.state != "queued":
            return
        self.state = "cancelled"
        self.add_class("cancelled")
        try:
            status = self.query_one(".queued-status", Horizontal)
            status.query_one(".queued-cancel", Static).remove()
            status.query_one(".queued-hint", Static).update("已取消")
        except NoMatches:
            pass  # Not composed yet.


class ChatWidget(VerticalScroll):
    """Displays the conversation as a single scrolling flow."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Full logical message list (display order), including messages
        # currently pruned from the DOM (old-message paging).
        self.messages: list[
            Static | Markdown | QueuedMessage | ThinkingBlock | ToolCard
        ] = []
        # How many of the oldest messages are pruned from the DOM.
        self._pruned_count = 0
        # Tool cards keyed by tool-call id; ToolCallRequest events may arrive
        # twice for the same call (once streamed, once before execution).
        self.tool_cards: dict[str, ToolCard] = {}
        # Queued steer messages keyed by steer-item id (RFC LIM-20).
        self.queued_cards: dict[str, QueuedMessage] = {}
        self._current_assistant: Markdown | None = None
        self._current_thinking: ThinkingBlock | None = None
        # Streaming batch buffers (flushed at _FLUSH_INTERVAL).
        self._assistant_buffer = ""
        self._thinking_dirty = False
        self._flush_timer: Any = None
        # Re-entrancy guard for concurrent flush paths (timer tick vs
        # explicit flush_stream): two concurrent Markdown.append calls are
        # the block-duplication bug this widget was built to avoid.
        self._flush_in_flight = False
        # Dedupes the "retry after mount" re-schedule of _prune so a burst
        # of messages cannot stack unlimited retry callbacks.
        self._prune_scheduled = False
        # Scroll-follow state: when the user scrolls up, new content no longer
        # drags the view down; a "back to bottom" pill appears instead (P2-4).
        self._follow = True
        self._pending_count = 0

    def compose(self) -> ComposeResult:
        # In-flow "load more" pill: sits above the oldest mounted message
        # while older messages are pruned (old-message paging).
        load_more = Static("", id="load-more", markup=False)
        load_more.display = False
        yield load_more
        # Docked to the bottom edge of the scroll view; hidden until needed.
        pill = Static("", id="back-to-bottom", markup=False)
        pill.display = False
        yield pill

    def on_mount(self) -> None:
        # Streaming flush cadence: a no-op tick is microseconds, so the
        # timer stays on — the last buffered chunk is rendered within one
        # interval of the stream ending.
        self._flush_timer = self.set_interval(_FLUSH_INTERVAL, self._tick_flush)

    def on_unmount(self) -> None:
        # Stop the timer explicitly so a re-mount never stacks a second one.
        if self._flush_timer is not None:
            self._flush_timer.stop()

    # -- streaming flush ----------------------------------------------------

    async def _tick_flush(self) -> None:
        """One time slice: at most one append + one thinking refresh + one scroll."""
        if self._flush_in_flight:
            return
        self._flush_in_flight = True
        try:
            scrolled = False
            if self._assistant_buffer and self._current_assistant is not None:
                buf, self._assistant_buffer = self._assistant_buffer, ""
                await self._current_assistant.append(buf)
                scrolled = True
            if self._thinking_dirty and self._current_thinking is not None:
                self._current_thinking.refresh_display()
                self._thinking_dirty = False
                scrolled = True
            if scrolled:
                self._auto_scroll()
        finally:
            self._flush_in_flight = False

    # -- messages -----------------------------------------------------------

    def add_user_message(
        self, text: str, attachments: list[Attachment] | None = None
    ) -> None:
        self._current_assistant = None
        self._current_thinking = None
        msg = Static(f"❯ {text}", classes="user-message", markup=False)
        self.messages.append(msg)
        self._mount_and_scroll(msg)
        if attachments:
            chips = []
            for attachment in attachments:
                label = "图片" if attachment.kind == "image" else "文件"
                chip = f"{label}: {attachment.name}"
                # Restored sessions may reference deleted files.
                if not Path(attachment.path).exists():
                    chip += "（已过期）"
                chips.append(chip)
            chip_msg = Static(
                "📎 " + " · ".join(chips), classes="attachment-chips", markup=False
            )
            self.messages.append(chip_msg)
            self._mount_and_scroll(chip_msg)

    # -- queued steer messages (LIM-20) --------------------------------------

    def add_queued_message(
        self, item_id: str, text: str, attachments: list[Attachment] | None = None
    ) -> QueuedMessage:
        """Optimistically render a message queued for steer injection."""
        self._current_assistant = None
        self._current_thinking = None
        card = QueuedMessage(item_id, text, attachments)
        self.queued_cards[item_id] = card
        self.messages.append(card)
        self._mount_and_scroll(card)
        return card

    def mark_steer_delivered(self, item_id: str) -> None:
        """Flip the queued card bound to ``item_id`` to a normal message."""
        card = self.queued_cards.pop(item_id, None)
        if card is not None:
            card.mark_delivered()

    def mark_steer_cancelled(self, item_id: str) -> None:
        """Flip the queued card bound to ``item_id`` to the cancelled state."""
        card = self.queued_cards.pop(item_id, None)
        if card is not None:
            card.mark_cancelled()

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

        Rendered as a collapsible summary (collapsed by default); the full
        text is materialized only while expanded. Updates are throttled to
        the flush cadence.
        """
        if self._current_thinking is None:
            # Thinking after assistant text starts a new block.
            self._current_assistant = None
            block = ThinkingBlock()
            self.messages.append(block)
            self._current_thinking = block
            await self.mount(block)
            self._prune()
        self._current_thinking.append(text)
        self._thinking_dirty = True

    async def append_assistant_text(self, text: str) -> None:
        """Append a streamed chunk to the current assistant Markdown block.

        Chunks accumulate in a buffer flushed at the flush cadence, so a
        fast chunk burst performs one Markdown re-render per slice instead
        of one per chunk (Markdown.append re-renders the whole document).
        """
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
            self._prune()
        self._assistant_buffer += text

    async def flush_stream(self) -> None:
        """Render any buffered streaming output immediately (tests).

        The 12 fps flush timer normally handles this within one interval;
        this is an escape hatch for callers that need the text on screen
        right now (Markdown.append is async and must be awaited). A no-op
        (empty buffers) never scrolls, and a flush already in flight is
        skipped to avoid concurrent Markdown.append calls.
        """
        if self._flush_in_flight:
            return
        self._flush_in_flight = True
        try:
            scrolled = False
            if self._assistant_buffer and self._current_assistant is not None:
                buf, self._assistant_buffer = self._assistant_buffer, ""
                await self._current_assistant.append(buf)
                scrolled = True
            if self._thinking_dirty and self._current_thinking is not None:
                self._current_thinking.refresh_display()
                self._thinking_dirty = False
                scrolled = True
            if scrolled:
                self._auto_scroll()
        finally:
            self._flush_in_flight = False

    def add_assistant_message(self, text: str) -> None:
        """Add a complete (non-streamed) assistant Markdown block."""
        self._current_assistant = None
        md = Markdown(text, classes="assistant-message")
        self.messages.append(md)
        self._mount_and_scroll(md)

    def clear(self) -> None:
        """Remove all rendered messages and tool cards (keeps the floaters)."""
        for child in list(self.children):
            if child.id not in ("back-to-bottom", "load-more"):
                child.remove()
        self.messages.clear()
        self.tool_cards.clear()
        self.queued_cards.clear()
        self._current_assistant = None
        self._current_thinking = None
        self._assistant_buffer = ""
        self._thinking_dirty = False
        self._pruned_count = 0
        self._pending_count = 0
        self._follow = True
        self._update_floater()
        self._update_load_more()

    # -- tool cards -----------------------------------------------------------

    def add_tool_card(
        self,
        tool_id: str,
        name: str,
        arguments: dict[str, Any],
        *,
        agent_owned: bool = True,
    ) -> ToolCard:
        """Get or create the card for a tool call (idempotent by tool-call id).

        ``agent_owned=False`` marks cards that do not back an agent tool call
        (user ``!`` bang commands): they are excluded from
        ``cancel_running_tool_cards()`` on turn interrupt.
        """
        existing = self.tool_cards.get(tool_id)
        if existing is not None:
            return existing
        # Text after a tool card must start a new assistant block.
        self._current_assistant = None
        self._current_thinking = None
        card = ToolCard(tool_id, name, arguments, agent_owned=agent_owned)
        self.tool_cards[tool_id] = card
        # Cards participate in old-message paging like any other message,
        # so the DOM bound covers them too (they are the bulk of a long
        # tool-heavy session). The dict keeps its reference for status
        # updates; toggle/cancel skip pruned (unmounted) cards.
        self.messages.append(card)
        self._mount_and_scroll(card)
        return card

    def toggle_tool_bodies(self) -> None:
        """Expand/collapse all tool cards that have output."""
        for card in self.tool_cards.values():
            if card.is_attached:
                card.toggle()

    def cancel_running_tool_cards(self) -> None:
        """Mark still-running agent-owned cards as interrupted (RFC LIM-53).

        Covers cards created from partial tool-call events during a stream
        that was then interrupted — those calls never execute, so no
        ToolResultEvent will ever arrive for them. Non-agent-owned cards
        (user bang commands) are skipped: their processes keep running and
        their results still arrive.
        """
        for card in self.tool_cards.values():
            # set_cancelled only mutates instance state + refreshes the
            # header (NoMatches-safe), so pruned/unmounted cards work too.
            if card.state == "running" and card.agent_owned:
                card.set_cancelled()

    # -- old-message paging --------------------------------------------------

    def _prune(self) -> None:
        """Keep at most ``_MAX_DOM_MESSAGES`` messages mounted.

        The oldest mounted messages are detached from the DOM (kept in
        ``self.messages`` for paging back). Widgets that have not finished
        their first mount are never pruned: removing them mid-compose makes
        Textual drop their composed children while still dispatching Mount,
        which crashes any on_mount that queries them. While the user is
        scrolled away from the tail, the viewport is held steady after the
        prune by compensating the scroll offset by the removed height, and
        messages currently inside the viewport are left alone.
        """
        if len(self.messages) - self._pruned_count <= _MAX_DOM_MESSAGES:
            self._update_load_more()
            return
        excess = len(self.messages) - self._pruned_count - _MAX_DOM_MESSAGES
        # How many are safe to remove: skip widgets still mounting, and skip
        # anything inside the current viewport when reading history.
        # region is screen-relative; overlap with self.region = "visible".
        to_remove = 0
        saw_unmounted = False
        while to_remove < excess:
            oldest = self.messages[self._pruned_count + to_remove]
            if not oldest._is_mounted:
                saw_unmounted = True
                break
            if not self._follow and oldest.region.overlaps(self.region):
                break  # inside the user's current view: leave it
            to_remove += 1
        if to_remove == 0:
            self._update_load_more()
            if excess > 0 and saw_unmounted and not self._prune_scheduled:
                # Every candidate is still mounting (a burst of messages in
                # one event-loop turn): retry once they have composed.
                self._prune_scheduled = True
                self.call_after_refresh(self._prune)
            return
        # Remove and accumulate the removed height directly: compensation
        # must not depend on a "stay" widget — a second prune in the same
        # refresh window removes it too, invalidating its region.
        removed_height = 0
        for _ in range(to_remove):
            removed_height += max(0, self.messages[self._pruned_count].region.height)
            self.messages[self._pruned_count].remove()
            self._pruned_count += 1
        if not self._follow and removed_height > 0:
            def _hold_viewport() -> None:
                self.scroll_y = max(0, self.scroll_y - removed_height)

            self.call_after_refresh(_hold_viewport)
        self._update_load_more()
        self._prune_scheduled = False
        # Skipped candidates (still mounting) get another pass once composed.
        if saw_unmounted and not self._prune_scheduled:
            self._prune_scheduled = True
            self.call_after_refresh(self._prune)

    def _load_more(self) -> None:
        """Restore the next page of pruned messages above the current ones."""
        if self._pruned_count == 0:
            return
        anchor = self.messages[self._pruned_count]
        anchor_y = anchor.region.y
        n = min(_PAGE_SIZE, self._pruned_count)
        batch = self.messages[self._pruned_count - n : self._pruned_count]
        self._pruned_count -= n
        for widget in batch:
            self.mount(widget, before=anchor)
        # Layout is stale until the next refresh; hold the viewport steady:
        # the anchor moved down by the loaded height, so scroll down by that.
        def _hold_viewport() -> None:
            self.scroll_y = min(
                self.max_scroll_y, self.scroll_y + (anchor.region.y - anchor_y)
            )

        self.call_after_refresh(_hold_viewport)
        self._update_load_more()
        self._prune_scheduled = False

    def _update_load_more(self) -> None:
        try:
            pill = self.query_one("#load-more", Static)
        except NoMatches:
            return  # Not composed yet.
        if self._pruned_count > 0:
            pill.update(f"↑ 加载更早 {self._pruned_count} 条消息")
            pill.display = True
        else:
            pill.display = False

    # -- helpers --------------------------------------------------------------

    def transcript_text(self) -> str:
        """Combined plain text of all text-like messages (for tests/debug)."""
        parts: list[str] = []
        for msg in self.messages:
            if isinstance(msg, Markdown):
                parts.append(msg.source)
            elif isinstance(msg, QueuedMessage):
                parts.append(msg.transcript())
            elif isinstance(msg, ThinkingBlock):
                parts.append(msg.content)
            elif isinstance(msg, ToolCard):
                continue  # tool cards are UI chrome, not conversation text
            else:
                parts.append(str(msg.content))
        return "\n".join(parts)

    def _mount_and_scroll(self, widget: Any) -> None:
        self.mount(widget)
        self._auto_scroll()
        self._prune()

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
        widget_id = getattr(event.widget, "id", None)
        if widget_id == "back-to-bottom":
            self._pending_count = 0
            self.scroll_end(animate=True)
        elif widget_id == "load-more":
            self._load_more()
