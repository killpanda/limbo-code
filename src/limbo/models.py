"""Shared data models for Limbo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    """A chat message in the conversation."""

    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
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
    """Result of executing a tool."""

    success: bool
    output: str | None = None
    error: str | None = None
    requires_confirmation: bool = False


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
