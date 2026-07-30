"""Tests for chat scroll-follow behavior (RFC P2-4) and structured errors (P1-2)."""

from __future__ import annotations

import pytest
from textual.app import App

from limbo.ui.widgets.chat import ChatWidget


class ChatApp(App[None]):
    def compose(self):
        yield ChatWidget(id="chat")


@pytest.mark.asyncio
async def test_new_content_scrolls_when_following():
    app = ChatApp()
    async with app.run_test(size=(60, 10)) as pilot:
        chat = app.query_one(ChatWidget)
        for i in range(30):
            chat.add_info(f"line {i}")
        await pilot.pause()
        assert chat._follow is True
        assert chat.scroll_offset.y >= chat.max_scroll_y - 2
        pill = chat.query_one("#back-to-bottom")
        assert pill.display is False


@pytest.mark.asyncio
async def test_scroll_up_pauses_follow_and_shows_pill():
    app = ChatApp()
    async with app.run_test(size=(60, 10)) as pilot:
        chat = app.query_one(ChatWidget)
        for i in range(30):
            chat.add_info(f"line {i}")
        await pilot.pause()

        chat.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert chat._follow is False

        chat.add_info("new message 1")
        chat.add_info("new message 2")
        await pilot.pause()
        # View stays put; the pill reports the pending messages.
        assert chat.scroll_offset.y < 2
        pill = chat.query_one("#back-to-bottom")
        assert pill.display is True
        assert "2 条新消息" in str(pill.content)

        # Scrolling back to the bottom resumes follow and hides the pill.
        chat.scroll_end(animate=False)
        await pilot.pause()
        assert chat._follow is True
        assert pill.display is False


@pytest.mark.asyncio
async def test_error_message_is_structured():
    app = ChatApp()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatWidget)
        chat.add_error("LLM 请求失败：HTTP 429\n详情请查看日志")
        await pilot.pause()
        text = chat.transcript_text()
        assert "✗ LLM 请求失败：HTTP 429" in text
        assert "  详情请查看日志" in text


@pytest.mark.asyncio
async def test_clear_keeps_floater_and_resets_follow():
    app = ChatApp()
    async with app.run_test(size=(60, 10)) as pilot:
        chat = app.query_one(ChatWidget)
        for i in range(30):
            chat.add_info(f"line {i}")
        await pilot.pause()
        chat.scroll_to(y=0, animate=False)
        await pilot.pause()
        chat.clear()
        await pilot.pause()
        assert chat._follow is True
        assert chat.query("#back-to-bottom")
        assert chat.messages == []
