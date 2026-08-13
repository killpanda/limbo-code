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
    # Commands that rewrite session/history state must NOT run mid-turn
    # (RFC LIM-20): they are rejected while the agent is busy. Read-only
    # commands opt in to busy availability here.
    allow_when_busy: bool = False
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
        """Look up a built-in by name (case-insensitive, '/'-carrying)."""
        return self._commands.get(name.lower())

    def all(self) -> list[SlashCommand]:
        return list(self._commands.values())

    def resolve(self, name: str, skills: list[Skill]) -> SlashCommand | Skill | None:
        """Resolve '/'-leading input to a built-in command or a skill.

        The single collision-policy site for both the autocomplete menu
        (``candidates``) and dispatch: a registered built-in always wins
        over a same-named skill. Built-in lookup is case-insensitive;
        skill names match exactly.
        """
        command = self.get(name)
        if command is not None:
            return command
        bare = name.removeprefix("/")
        for skill in skills:
            if skill.name == bare:
                return skill
        return None

    def candidates(self, skills: list[Skill]) -> list[SlashCommand]:
        """Built-in commands plus skills (built-ins win on name collisions)."""
        candidates = self.all()
        for skill in skills:
            # Same policy as resolve(): a built-in claims the name first.
            if self.get(f"/{skill.name}") is not None:
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
