"""Shared data models for Limbo."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class Attachment(BaseModel):
    """A file/image attached to a user message (paste of clipboard content).

    ``path`` points at the stored content (clipboard images are saved under
    ``~/.limbo/attachments/`` at paste time; file attachments reference the
    original file). Only the path is persisted in sessions — never bytes —
    so restored sessions may reference files that no longer exist; readers
    must tolerate that (clients skip missing images, the UI marks them
    expired).

    Persistence trade-off (deliberate, LIM-17 review): only image
    attachments that pass the vision gate are stored on ``Message`` — file
    attachments are folded into ``content`` (inline text or path
    reference), so after a session resume the model loses nothing but the
    chat no longer renders a chip for them.
    """

    kind: str  # "image" | "file"
    name: str
    path: str
    mime: str | None = None


class Message(BaseModel):
    """A chat message in the conversation.

    ``id`` makes each message an addressable entry (pi-style entry model):
    compaction records reference the first kept entry by id. Older session
    files without ids load fine — the default factory mints one on parse.
    """

    id: str = Field(default_factory=lambda: secrets.token_hex(8))
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Image attachments to send as multimodal content blocks (only set when
    # the current model supports vision; otherwise agent.py degrades them
    # to path references in ``content`` and leaves this None).
    attachments: list[Attachment] | None = None
    # Reasoning/thinking text produced by reasoning models. Stored so it can
    # be replayed to APIs that require it (e.g. Kimi K3) on later turns.
    reasoning: str | None = None
    # Signature authenticating a replayed thinking block (Anthropic dialect).
    reasoning_signature: str | None = None


class ToolCall(BaseModel):
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result of executing a tool.

    ``attachments`` carries image payloads a tool wants the model to see
    (e.g. read returning an image file). Only the path is persisted —
    never bytes — same policy as user-message attachments, so restored
    sessions may reference files that no longer exist; clients skip
    missing images when replaying.
    """

    success: bool
    output: str | None = None
    error: str | None = None
    attachments: list[Attachment] | None = None


@dataclass(frozen=True)
class TextChunk:
    text: str


@dataclass(frozen=True)
class ThinkingChunk:
    """A streamed reasoning/thinking delta from a reasoning model."""

    text: str
    # Thinking-block signature (Anthropic dialect); carried on an empty-text
    # chunk at the end of a thinking block.
    signature: str | None = None


@dataclass(frozen=True)
class ToolCallEvent:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class CompletionMeta:
    """Response-level metadata emitted once per LLM call, after all chunks.

    Carries whatever the provider returned: token usage (including cache-hit
    counters where the provider exposes them), the finish/stop reason, and
    client-measured timing. All fields may be None — providers differ widely
    in what they report.
    """

    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    # Seconds from request start to the first streamed chunk.
    ttft: float | None = None
    # Seconds from request start to the end of the stream.
    duration: float | None = None


LLMEvent = TextChunk | ThinkingChunk | ToolCallEvent | CompletionMeta
