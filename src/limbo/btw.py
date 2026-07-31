"""Side questions (/btw): a one-shot LLM query over a history snapshot.

/btw lets the user ask a quick question mid-task without touching the main
conversation (Claude Code parity): the query runs on a *copy* of the
history with a side-question instruction appended to the system prompt, no
tools, a single streaming call — and the answer is shown in a transient
overlay and then discarded. Nothing is written back to the session history
or token totals.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from limbo.llm.client import LLMClient, RequestHook
from limbo.models import Message, TextChunk

# Appended to the session's own system prompt: keeps full project context
# but constrains the answer to a direct one-shot reply (no tools, no
# "let me try...", no promises of action — the answer cannot affect the
# main conversation).
BTW_INSTRUCTION = (
    "你现在是一个轻量级的“侧问”实例：用户正在主会话中进行一个更大的任务，"
    "这只是任务中途的一个快速问题。\n"
    "规则：\n"
    "1. 基于对话上下文直接、简洁地回答，一次性答完。\n"
    "2. 你没有任何工具可用，也不要承诺采取任何行动"
    "（不要说“我来试试”“让我查一下”）。\n"
    "3. 你的回答不会进入主会话历史，仅临时展示给用户。"
)


def build_btw_messages(history: list[Message], question: str) -> list[Message]:
    """Construct the one-shot side-query prompt from a history snapshot.

    The leading system message gets the side-question instruction appended
    (via a copy — the session's own system message is never mutated); the
    rest of the history is reused as-is, and the question goes last as a
    user message. ``history`` itself is not modified.
    """
    messages: list[Message] = []
    if history and history[0].role == "system":
        system = history[0]
        messages.append(
            system.model_copy(
                update={
                    "content": (system.content or "") + "\n\n" + BTW_INSTRUCTION
                }
            )
        )
        messages.extend(history[1:])
    else:
        messages.append(Message(role="system", content=BTW_INSTRUCTION))
        messages.extend(history)
    messages.append(Message(role="user", content=question))
    return messages


async def btw_query(
    llm_client: LLMClient,
    history: list[Message],
    question: str,
    on_request: RequestHook | None = None,
) -> AsyncIterator[str]:
    """Stream the side-question answer as text chunks (one call, no tools).

    Thinking chunks and completion metadata are intentionally dropped:
    thinking is noise for a quick overlay answer, and side questions do not
    count toward the session's token totals. Client exceptions propagate —
    the UI layer maps them to a friendly message.
    """
    messages = build_btw_messages(history, question)
    async for event in llm_client.chat(messages, tools=[], on_request=on_request):
        if isinstance(event, TextChunk):
            yield event.text
