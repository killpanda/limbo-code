"""/btw overlay: a transient side-question answer that never enters the session."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown, Static

from limbo.btw import btw_query
from limbo.llm.client import LLMClient
from limbo.llm.retry import friendly_message
from limbo.models import Message
from limbo.trace import TraceLogger

FOOTER_BUSY = "回答中… · Esc 关闭（答案不会进入会话）"
FOOTER_DONE = "Esc 关闭（答案不会进入会话）"


class BtwScreen(ModalScreen[None]):
    """Floating side-question overlay.

    Streams a one-shot answer over a snapshot of the conversation. The
    underlying main screen keeps running (same discipline as the 2048
    modal), and closing discards the answer — nothing is written back to
    the session history, the steer queue, or the token totals.
    """

    BINDINGS = [
        Binding("escape", "close", "关闭"),
        Binding("q", "close", "关闭", show=False),
    ]

    def __init__(
        self,
        question: str,
        llm_client: LLMClient,
        history: list[Message],
        trace: TraceLogger | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._question = question
        self._llm_client = llm_client
        self._history = history
        self._trace = trace

    def compose(self) -> ComposeResult:
        with Vertical(id="btw-container"):
            yield Static(
                f"💬 btw · {self._question}", id="btw-question", markup=False
            )
            with VerticalScroll(id="btw-answer-scroll"):
                yield Markdown(id="btw-answer")
            yield Static(FOOTER_BUSY, id="btw-footer", markup=False)

    def on_mount(self) -> None:
        self.run_worker(self._stream_answer())

    def action_close(self) -> None:
        self.dismiss()

    async def _stream_answer(self) -> None:
        footer = self.query_one("#btw-footer", Static)
        answer = self.query_one("#btw-answer", Markdown)

        def _trace_request(body: dict) -> None:
            if self._trace is not None:
                self._trace.log("llm_request", kind="btw", body=body)

        answered = ""
        try:
            async for chunk in btw_query(
                self._llm_client,
                self._history,
                self._question,
                on_request=_trace_request,
            ):
                answered += chunk
                # Awaited per chunk, same rationale as ChatWidget: unawaited
                # appends queue stale re-renders (visually duplicated text).
                await answer.append(chunk)
        except Exception as e:  # noqa: BLE001 - overlay must never crash the app
            hint = friendly_message(e)
            note = hint or f"出错了：{e}"
            await answer.append(f"\n\n⚠ {note}" if answered else f"⚠ {note}")
        if self._trace is not None:
            self._trace.log("btw_response", chars=len(answered))
        footer.update(FOOTER_DONE)
