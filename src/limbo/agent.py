"""Agent conversation loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from limbo.config import Config
from limbo.llm.client import LLMClient
from limbo.models import Message, TextChunk, ToolCallEvent, ToolResult
from limbo.tools.registry import ToolRegistry


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResultEvent:
    id: str
    name: str
    result: ToolResult
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ErrorEvent:
    message: str


AgentEvent = TextDelta | ToolCallRequest | ToolResultEvent | ErrorEvent


class Agent:
    """Orchestrates the conversation between user, LLM, and tools."""

    def __init__(self, config: Config, llm_client: LLMClient, workdir: Path):
        self.config = config
        self.llm_client = llm_client
        self.workdir = workdir
        self.registry = ToolRegistry(workdir=workdir)
        self.messages: list[Message] = []
        self._pending_tool: dict[str, Any] | None = None
        self._init_system_message()

    def _init_system_message(self) -> None:
        self.messages.append(
            Message(
                role="system",
                content=(
                    "You are an expert coding assistant operating inside limbo, "
                    "a coding agent harness. You help users by reading files, "
                    "executing commands, editing code, and writing new files.\n\n"
                    "Available tools:\n"
                    "- read: Read file contents\n"
                    "- bash: Execute bash commands\n"
                    "- edit: Make surgical edits to files (find exact text and replace)\n"
                    "- write: Create or overwrite files\n"
                    "- grep: Search file contents for patterns (respects .gitignore)\n"
                    "- find: Find files by glob pattern (respects .gitignore)\n"
                    "- ls: List directory contents\n\n"
                    "Guidelines:\n"
                    "- Prefer grep/find/ls tools over bash for file exploration\n"
                    "- Use read to examine files before editing\n"
                    "- Use edit for precise changes (old_text must match exactly)\n"
                    "- Use write only for new files or complete rewrites\n"
                    "- Be concise in your responses\n"
                    "- Show file paths clearly when working with files"
                ),
            )
        )

    def run(self, user_input: str) -> Iterator[AgentEvent]:
        self.messages.append(Message(role="user", content=user_input))

        for _ in range(self.config.llm.max_iterations):
            try:
                yield from self._call_llm()
            except Exception as e:  # noqa: BLE001
                yield ErrorEvent(message=f"LLM error: {e}")
                return

            last = self.messages[-1]
            if not last.tool_calls:
                break

            for tc in last.tool_calls:
                name = tc["function"]["name"]
                arguments = tc["function"]["arguments"]
                yield ToolCallRequest(id=tc["id"], name=name, arguments=arguments)
                result = self.registry.execute(name, arguments, dry_run=True)
                yield ToolResultEvent(
                    id=tc["id"], name=name, result=result, arguments=arguments
                )
                if result.requires_confirmation:
                    self._pending_tool = tc
                    return
                self.messages.append(
                    Message(
                        role="tool",
                        content=result.output or result.error or "",
                        tool_call_id=tc["id"],
                    )
                )

    def apply_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = self.registry.execute(name, arguments, dry_run=False)
        last = self.messages[-1]
        for tc in last.tool_calls or []:
            if (
                tc["function"]["name"] == name
                and tc["function"]["arguments"] == arguments
            ):
                self.messages.append(
                    Message(
                        role="tool",
                        content=result.output or result.error or "",
                        tool_call_id=tc["id"],
                    )
                )
                break
        self._pending_tool = None
        return result

    def _call_llm(self) -> Iterator[AgentEvent]:
        tool_definitions = self.registry.definitions()
        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []

        for event in self.llm_client.chat(self.messages, tools=tool_definitions):
            if isinstance(event, TextChunk):
                assistant_content += event.text
                yield TextDelta(text=event.text)
            elif isinstance(event, ToolCallEvent):
                tool_calls.append(
                    {
                        "id": event.id,
                        "type": "function",
                        "function": {
                            "name": event.name,
                            "arguments": event.arguments,
                        },
                    }
                )
                yield ToolCallRequest(
                    id=event.id, name=event.name, arguments=event.arguments
                )

        self.messages.append(
            Message(
                role="assistant",
                content=assistant_content or None,
                tool_calls=tool_calls if tool_calls else None,
            )
        )
