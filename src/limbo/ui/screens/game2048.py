"""2048 mini-game: pure game logic + a modal screen.

The screen is a modal pushed on top of the main chat screen. Textual keeps
workers on the underlying screen running while a modal is up, so the agent
turn continues in the background — the game never interrupts it.
"""

from __future__ import annotations

import random
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

Direction = Literal["up", "down", "left", "right"]

TARGET = 2048


class Game2048:
    """Pure 2048 game logic, independent of the UI."""

    SIZE = 4

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self.grid: list[list[int]] = [
            [0] * self.SIZE for _ in range(self.SIZE)
        ]
        self.score = 0
        self.won = False
        self.spawn()
        self.spawn()

    # -- state ----------------------------------------------------------------

    @property
    def over(self) -> bool:
        """Game over when no empty cell and no adjacent equal tiles remain."""
        size = self.SIZE
        for r in range(size):
            for c in range(size):
                if self.grid[r][c] == 0:
                    return False
                if c + 1 < size and self.grid[r][c] == self.grid[r][c + 1]:
                    return False
                if r + 1 < size and self.grid[r][c] == self.grid[r + 1][c]:
                    return False
        return True

    def spawn(self) -> bool:
        """Add a 2 (90%) or 4 (10%) to a random empty cell."""
        empties = [
            (r, c)
            for r in range(self.SIZE)
            for c in range(self.SIZE)
            if self.grid[r][c] == 0
        ]
        if not empties:
            return False
        r, c = self._rng.choice(empties)
        self.grid[r][c] = 4 if self._rng.random() < 0.1 else 2
        return True

    # -- moves ------------------------------------------------------------------

    def move(self, direction: Direction) -> bool:
        """Slide and merge the grid. Returns True if anything changed."""
        lines = self._lines(direction)
        gained = 0
        new_lines = []
        for line in lines:
            merged, line_gained = self._merge_left(line)
            new_lines.append(merged)
            gained += line_gained
        if new_lines == lines:
            return False
        self.score += gained
        self._write_lines(direction, new_lines)
        if not self.won and any(
            value >= TARGET for row in self.grid for value in row
        ):
            self.won = True
        self.spawn()
        return True

    def _lines(self, direction: Direction) -> list[list[int]]:
        """Extract rows/columns so every move reduces to 'merge left'."""
        size = self.SIZE
        if direction == "left":
            return [row[:] for row in self.grid]
        if direction == "right":
            return [row[::-1] for row in self.grid]
        if direction == "up":
            return [[self.grid[r][c] for r in range(size)] for c in range(size)]
        return [
            [self.grid[r][c] for r in range(size - 1, -1, -1)]
            for c in range(size)
        ]

    def _write_lines(self, direction: Direction, lines: list[list[int]]) -> None:
        """Inverse of `_lines`."""
        size = self.SIZE
        if direction == "left":
            self.grid = [line[:] for line in lines]
        elif direction == "right":
            self.grid = [line[::-1] for line in lines]
        elif direction == "up":
            for c in range(size):
                for r in range(size):
                    self.grid[r][c] = lines[c][r]
        else:
            for c in range(size):
                for r in range(size):
                    self.grid[size - 1 - r][c] = lines[c][r]

    @staticmethod
    def _merge_left(line: list[int]) -> tuple[list[int], int]:
        """Compress and merge one line towards index 0."""
        tiles = [value for value in line if value]
        merged: list[int] = []
        gained = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged.append(tiles[i] * 2)
                gained += tiles[i] * 2
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        return merged + [0] * (len(line) - len(merged)), gained


class Game2048Screen(ModalScreen[None]):
    """Playable 2048 modal. The agent keeps running underneath."""

    BINDINGS = [
        Binding("up", "move('up')", "↑"),
        Binding("down", "move('down')", "↓"),
        Binding("left", "move('left')", "←"),
        Binding("right", "move('right')", "→"),
        Binding("w", "move('up')", show=False),
        Binding("s", "move('down')", show=False),
        Binding("a", "move('left')", show=False),
        Binding("d", "move('right')", show=False),
        Binding("r", "restart", "重开"),
        Binding("q", "close", "关闭"),
        Binding("escape", "close", "关闭"),
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.game = Game2048()

    def compose(self) -> ComposeResult:
        with Vertical(id="game-container"):
            yield Static("", id="game-header", markup=False)
            with Grid(id="game-grid"):
                for r in range(self.game.SIZE):
                    for c in range(self.game.SIZE):
                        yield Static("", classes="tile", id=f"cell-{r}-{c}")
            yield Static(
                "方向键 / WASD 移动 · R 重开 · Q/Esc 关闭",
                id="game-footer",
                markup=False,
            )

    def on_mount(self) -> None:
        self._refresh()

    # -- actions --------------------------------------------------------------

    def action_move(self, direction: Direction) -> None:
        if self.game.over:
            return
        if self.game.move(direction):
            self._refresh()

    def action_restart(self) -> None:
        self.game = Game2048()
        self._refresh()

    def action_close(self) -> None:
        self.dismiss()

    # -- rendering --------------------------------------------------------------

    def _refresh(self) -> None:
        header = self.query_one("#game-header", Static)
        status = ""
        if self.game.over:
            status = " · 游戏结束，按 R 重开"
        elif self.game.won:
            status = " · 🎉 达成 2048！可继续挑战"
        header.update(f"2048 · 分数 {self.game.score}{status}")

        for r in range(self.game.SIZE):
            for c in range(self.game.SIZE):
                value = self.game.grid[r][c]
                cell = self.query_one(f"#cell-{r}-{c}", Static)
                cell.update(str(value) if value else "")
                if value:
                    tile_class = (
                        f"tile-{value}" if value <= TARGET else "tile-super"
                    )
                    cell.set_classes(f"tile {tile_class}")
                else:
                    cell.set_classes("tile")
