import pytest
from textual.app import App
from textual.widgets import Static

from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget, UserSubmitted


@pytest.mark.asyncio
async def test_chat_adds_messages():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        widget.add_user_message("hi")
        await widget.append_assistant_text("hello")
        assert len(widget.messages) == 2


@pytest.mark.asyncio
async def test_input_submits_event():
    submitted = []

    class TestApp(App[None]):
        def compose(self):
            yield InputWidget(id="input")

        def on_user_submitted(self, event: UserSubmitted) -> None:
            submitted.append(event)

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.text = "hello"
        widget.action_submit()

    assert len(submitted) == 1
    assert isinstance(submitted[0], UserSubmitted)
    assert submitted[0].message == "hello"


@pytest.mark.asyncio
async def test_input_submits_on_enter_key():
    submitted = []

    class TestApp(App[None]):
        def compose(self):
            yield InputWidget(id="input")

        def on_user_submitted(self, event: UserSubmitted) -> None:
            submitted.append(event)

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "hello"
        await pilot.press("enter")
        await pilot.pause()

    assert len(submitted) == 1
    assert submitted[0].message == "hello"


@pytest.mark.asyncio
async def test_input_shift_enter_inserts_newline():
    class TestApp(App[None]):
        def compose(self):
            yield InputWidget(id="input")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(InputWidget)
        widget.focus()
        widget.text = "hello"
        widget.action_cursor_line_end()
        await pilot.press("shift+enter")
        await pilot.pause()

    assert widget.text == "hello\n"


@pytest.mark.asyncio
async def test_chat_append_streams_into_one_block():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        await widget.append_assistant_text("[bold]chunk1[/bold]")
        await widget.append_assistant_text("[italic]chunk2[/italic]")
        await widget.flush_stream()  # render the batched buffer immediately

        # Both chunks accumulate in a single assistant block.
        assert len(widget.messages) == 1
        combined = widget.transcript_text()
        assert "[bold]chunk1[/bold]" in combined
        assert "[italic]chunk2[/italic]" in combined


@pytest.mark.asyncio
async def test_tool_card_dedupes_by_id():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        card1 = widget.add_tool_card("c1", "read", {"path": "a.py"})
        card2 = widget.add_tool_card("c1", "read", {"path": "a.py"})
        assert card1 is card2
        assert len(widget.tool_cards) == 1


@pytest.mark.asyncio
async def test_tool_card_state_transitions():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        card = widget.add_tool_card("c1", "write", {"path": "x.txt", "content": "hi"})
        assert card.state == "running"

        card.set_success("written")
        assert card.state == "success"

        err_card = widget.add_tool_card("c2", "read", {"path": "missing.txt"})
        err_card.set_error("not found")
        assert err_card.state == "error"


@pytest.mark.asyncio
async def test_tool_card_toggle_requires_body():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        card = widget.add_tool_card("c1", "ls", {})
        await pilot.pause()

        # No output yet: toggling is a no-op and no body exists.
        card.toggle()
        assert card.body is None

        card.set_success("file1\nfile2")
        card.toggle()
        assert card.body is not None and card.body.display is True
        card.toggle()
        # Collapsing destroys the lazy body: the card is one line again.
        assert card.body is None


@pytest.mark.asyncio
async def test_chat_streaming_burst_does_not_duplicate_blocks():
    """Regression: fast chunk bursts must not duplicate rendered blocks.

    Markdown.append() defers its re-render; firing many appends without
    awaiting each one queues stale renders that re-mount existing blocks.
    """

    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        text = (
            "First paragraph.\n\n"
            "Second paragraph.\n\n"
            "•  bullet one\n"
            "•  bullet two\n\n"
            "Final paragraph."
        )
        # Token-sized chunks with no pause between calls simulates a fast
        # streaming burst (e.g. several SSE events per network read).
        chunks = [text[i : i + 4] for i in range(0, len(text), 4)]
        for chunk in chunks:
            await widget.append_assistant_text(chunk)
        await widget.flush_stream()  # render the batched buffer immediately

        md = widget.messages[-1]
        rendered = "\n".join(
            str(getattr(child, "content", "")) for child in md.children
        )
        assert rendered.count("Second paragraph.") == 1
        assert rendered.count("bullet one") == 1
        assert rendered.count("Final paragraph.") == 1

@pytest.mark.asyncio
async def test_tool_card_run_code_shows_description_and_source():
    """run_code cards must be observable: header shows the program's
    description, and expanding shows the full program source (python)
    before the result — otherwise Code Mode work is a black box."""
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    code = "content = await tools.read(path='a.py')\nreturn content"
    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        card = widget.add_tool_card(
            "c1",
            "run_code",
            {"code": code, "description": "读 a.py 并返回内容"},
        )
        await pilot.pause()

        # The one-line header carries the program's intent.
        assert "读 a.py 并返回内容" in str(card.header.render())

        card.set_success("ok: file content")
        card.toggle()
        assert card.body is not None and card.body.display is True
        # Expanding shows the source first, then the result. (The body is a
        # hidden RichLog: in tests it stays size-unknown so writes land in
        # _deferred_renders and lines stays empty — assert on the sections
        # the card was told to render instead.)
        sections = card._body_content
        assert sections is not None
        assert sections[0] == (code, "python")
        assert sections[1] == ("ok: file content", None)

@pytest.mark.asyncio
async def test_thinking_collapsed_by_default_and_expandable():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        await widget.append_thinking_text("step one ")
        await widget.append_thinking_text("step two")
        await pilot.pause()

        block = widget.messages[-1]
        from limbo.ui.widgets.chat import ThinkingBlock
        assert isinstance(block, ThinkingBlock)
        # Collapsed by default: summary line only, body hidden.
        assert block.collapsed
        assert block._body.display is False
        summary = str(block._summary.render())
        assert "thinking" in summary and "字" in summary

        # Expand shows the full accumulated text.
        block.toggle()
        assert block._body.display is True
        assert "step one step two" in str(block._body.render())
        block.toggle()
        assert block.collapsed


@pytest.mark.asyncio
async def test_chat_prunes_old_messages_and_pages_back():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    from limbo.ui.widgets.chat import _MAX_DOM_MESSAGES, _PAGE_SIZE

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        for i in range(_MAX_DOM_MESSAGES + 60):
            widget.add_info(f"msg {i}")
        await pilot.pause()

        # DOM is bounded; the overflow is tracked for paging.
        mounted = [c for c in widget.children if c.id not in ("back-to-bottom", "load-more")]
        assert len(mounted) <= _MAX_DOM_MESSAGES + 2
        assert widget._pruned_count == 60

        # The load-more pill is shown while older messages exist.
        pill = widget.query_one("#load-more")
        assert pill.display is True
        assert "加载更早" in str(pill.render())

        # Loading a page restores the oldest 50 (msg 10..59) and re-anchors.
        widget._load_more()
        await pilot.pause()
        assert widget._pruned_count == 60 - _PAGE_SIZE
        assert any(
            isinstance(c, Static) and str(c.render()) == "msg 10" for c in widget.children
        )
        assert not any(
            isinstance(c, Static) and str(c.render()) == "msg 0" for c in widget.children
        )
        # A second page brings the rest (msg 0..9) back too.
        widget._load_more()
        await pilot.pause()
        assert widget._pruned_count == 0
        assert any(
            isinstance(c, Static) and str(c.render()) == "msg 0" for c in widget.children
        )

        # The transcript always keeps the full history.
        assert "msg 0" in widget.transcript_text()
        assert f"msg {_MAX_DOM_MESSAGES + 59}" in widget.transcript_text()


