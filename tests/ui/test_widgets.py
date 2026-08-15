import pytest
from textual.app import App

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
        await pilot.pause()

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

        # No output yet: toggling is a no-op and body stays hidden.
        card.toggle()
        assert card.body.display is False

        card.set_success("file1\nfile2")
        card.toggle()
        assert card.body.display is True
        card.toggle()
        assert card.body.display is False


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
        await pilot.pause()

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
        assert card.body.display is True
        # Expanding shows the source first, then the result. (The body is a
        # hidden RichLog: in tests it stays size-unknown so writes land in
        # _deferred_renders and lines stays empty — assert on the sections
        # the card was told to render instead.)
        sections = card._body_content
        assert sections is not None
        assert sections[0] == (code, "python")
        assert sections[1] == ("ok: file content", None)
