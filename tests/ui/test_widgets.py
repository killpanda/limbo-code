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
