
import pytest
from textual.app import App

from limbo.ui.widgets.chat import ChatWidget


async def wait_until(pilot, predicate, attempts: int = 200) -> None:
    """Pause until predicate holds (timeout-safe), avoiding pause-count
    flakiness under suite load."""
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("condition not reached within timeout")



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


