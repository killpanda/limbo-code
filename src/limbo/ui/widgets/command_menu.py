"""Slash-command autocomplete menu.

Shown above the input box whenever the input starts with ``/`` (and contains
no whitespace yet). Enter executes the highlighted command (arg-taking
commands are completed into the input instead), Tab completes, Esc closes.
"""

from __future__ import annotations

from textual.widgets import OptionList
from textual.widgets.option_list import Option

from limbo.ui.commands import SlashCommand


class SlashCommandMenu(OptionList):
    """Autocomplete popup listing slash commands; hidden by default."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.display = False
        self._commands: list[SlashCommand] = []

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    def show_commands(self, commands: list[SlashCommand]) -> None:
        self._commands = list(commands)
        self.clear_options()
        self.add_options(
            [Option(f"{c.name}  {c.description}") for c in self._commands]
        )
        if self._commands:
            self.highlighted = 0
        self.display = True

    def close(self) -> None:
        self.display = False

    def highlighted_command(self) -> SlashCommand | None:
        index = self.highlighted
        if index is None or not (0 <= index < len(self._commands)):
            return None
        return self._commands[index]
