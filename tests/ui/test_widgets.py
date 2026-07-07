import pytest
from textual.app import App

from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.file_preview import FilePreviewWidget
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
        widget.add_assistant_text("hello")
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
async def test_file_preview_escapes_markup():
    class TestApp(App[None]):
        def compose(self):
            yield FilePreviewWidget(id="preview")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(FilePreviewWidget)
        widget.show("title", "[bold]not bold[/bold]")
        # The literal brackets should be escaped in the stored content string.
        assert r"\[bold]not bold\[/bold]" in widget.content


@pytest.mark.asyncio
async def test_chat_append_does_not_interpret_markup():
    class TestApp(App[None]):
        def compose(self):
            yield ChatWidget(id="chat")

    app = TestApp()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one(ChatWidget)
        widget.add_assistant_text("[bold]chunk1[/bold]")
        widget.append_assistant_text("[italic]chunk2[/italic]")

        last = widget.messages[-1]
        combined = str(last.content)
        assert "[bold]chunk1[/bold]" in combined
        assert "[italic]chunk2[/italic]" in combined
        # The rendered visual must contain the literal brackets, not styled text.
        assert "[bold]chunk1[/bold][italic]chunk2[/italic]" == last.visual.plain
