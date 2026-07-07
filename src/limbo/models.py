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
class ToolCallEvent:
    id: str
    name: str
    arguments: dict[str, Any]


LLMEvent = TextChunk | ToolCallEvent
