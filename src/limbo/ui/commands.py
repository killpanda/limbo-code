"""Slash-command registry: one source for menu metadata and dispatch.

A slash command is defined once — name, description, whether it takes
arguments, and its handler. The autocomplete menu renders from the registry
and the input dispatcher consults it, so the two can never drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from limbo.skills import Skill


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    takes_args: bool = False
    kind: str = "builtin"  # "builtin" | "skill"
    handler: Callable[[str], None] | None = field(
        default=None, compare=False, repr=False
    )


class SlashCommandRegistry:
    """Holds the built-in commands; merges skills for display and lookup."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def all(self) -> list[SlashCommand]:
        return list(self._commands.values())

    def candidates(self, skills: list[Skill]) -> list[SlashCommand]:
        """Built-in commands plus skills (built-ins win on name collisions)."""
        candidates = self.all()
        for skill in skills:
            if f"/{skill.name}" in self._commands:
                continue
            candidates.append(
                SlashCommand(
                    f"/{skill.name}",
                    f"[skill] {skill.description}".rstrip(),
                    takes_args=True,
                    kind="skill",
                )
            )
        return candidates

    def help_text(self) -> str:
        return "可用命令：" + " · ".join(
            f"{command.name} {command.description}" for command in self.all()
        )
