import pytest
from textual.app import App

from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.confirm import ConfirmDialog, Rejected
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
async def test_confirm_dialog_escape_posts_rejected():
    rejected = []

    class TestApp(App[None]):
        def on_rejected(self, event: Rejected) -> None:
            rejected.append(event)

    app = TestApp()
    async with app.run_test() as pilot:
        dialog = ConfirmDialog(title="Test", body="body")
        pilot.app.push_screen(dialog)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

    assert len(rejected) == 1
    assert isinstance(rejected[0], Rejected)


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

        card.set_pending("preview body")
        assert card.state == "pending"

        card.set_applied("written")
        assert card.state == "applied"

        card.set_rejected()
        assert card.state == "rejected"

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
