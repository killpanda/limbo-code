"""Tests for the 2048 mini-game: logic (Game2048) and UI (Game2048Screen)."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest

from limbo.config import Config
from limbo.models import TextChunk
from limbo.ui.app import LimboApp
from limbo.ui.screens.game2048 import Game2048, Game2048Screen
from limbo.ui.screens.main import MainScreen
from limbo.ui.widgets.chat import ChatWidget
from limbo.ui.widgets.input import InputWidget

# -- pure game logic ----------------------------------------------------------


def make_game(grid: list[list[int]]) -> Game2048:
    """A game with a fixed grid, bypassing the random initial spawns."""
    game = Game2048(rng=random.Random(0))
    game.grid = [row[:] for row in grid]
    game.score = 0
    game.won = False
    return game


def test_initial_state_has_two_tiles():
    game = Game2048(rng=random.Random(0))
    tiles = [v for row in game.grid for v in row if v]
    assert len(tiles) == 2
    assert all(v in (2, 4) for v in tiles)
    assert game.score == 0
    assert not game.won
    assert not game.over


def test_merge_left_merges_pairs_once():
    merged, gained = Game2048._merge_left([2, 2, 2, 2])
    assert merged == [4, 4, 0, 0]
    assert gained == 8

    merged, gained = Game2048._merge_left([4, 4, 8, 0])
    assert merged == [8, 8, 0, 0]
    assert gained == 8

    # A tile created by a merge must not merge again in the same move.
    merged, gained = Game2048._merge_left([2, 2, 4, 0])
    assert merged == [4, 4, 0, 0]
    assert gained == 4


def test_move_left_compresses_and_merges():
    game = make_game(
        [
            [2, 0, 2, 4],
            [0, 0, 0, 0],
            [2, 2, 2, 0],
            [8, 8, 8, 8],
        ]
    )
    assert game.move("left")
    # Row 0: [2,0,2,4] -> [4,4,0,0] (a spawned tile may also land on it)
    assert game.grid[0][0] == 4
    assert game.grid[0][1] == 4
    # Row 2: [2,2,2,0] -> [4,2,0,0]; row 3: [8,8,8,8] -> [16,16,0,0]
    assert game.grid[2][0] == 4
    assert game.grid[2][1] == 2
    assert game.grid[3][0] == 16
    assert game.grid[3][1] == 16
    assert game.score == 4 + 4 + 32


def test_move_right_up_down_symmetry():
    game = make_game(
        [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 4],
            [0, 0, 4, 4],
        ]
    )
    assert game.move("up")
    # Column 0: [2,2,0,0] -> 4; column 2: [0,0,0,4] -> 4;
    # column 3: [0,0,4,4] -> 8. All land in row 0 (spawn can't overwrite them).
    assert game.grid[0][0] == 4
    assert game.grid[0][2] == 4
    assert game.grid[0][3] == 8


def test_move_returns_false_when_nothing_changes():
    game = make_game(
        [
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert not game.move("left")
    assert game.score == 0


def test_move_spawns_new_tile():
    game = make_game(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    game.move("left")
    tiles = [v for row in game.grid for v in row if v]
    assert len(tiles) == 2  # merged 4 + one spawned tile


def test_game_over_detection():
    game = make_game(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
    )
    assert game.over

    game = make_game(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 2, 4],  # adjacent equal pair -> a move exists
        ]
    )
    assert not game.over


def test_win_detection():
    game = make_game(
        [
            [1024, 1024, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    game.move("left")
    assert game.won


# -- UI -----------------------------------------------------------------------


class FakeLLMClient:
    def __init__(self, responses=None, gate: asyncio.Event | None = None):
        self.responses = list(responses or [])
        self.gate = gate
        self.calls = 0

    async def chat(self, messages, tools, on_request=None):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.responses:
            for event in self.responses.pop(0):
                yield event


def make_app(tmp_path: Path, fake_llm=None) -> LimboApp:
    cfg = Config()
    cfg.llm.api_key = "test"
    return LimboApp(
        workdir=tmp_path,
        config=cfg,
        llm_client=fake_llm or FakeLLMClient(),
        session_dir=tmp_path / "sessions",
    )


async def submit(pilot, text: str) -> None:
    screen = pilot.app.screen_stack[-1]
    input_widget = screen.query_one("#input", InputWidget)
    input_widget.text = text
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.asyncio
async def test_slash_command_opens_game(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "/2048")
        assert isinstance(pilot.app.screen, Game2048Screen)

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(pilot.app.screen, MainScreen)


@pytest.mark.asyncio
async def test_ctrl_g_opens_game(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(pilot.app.screen, Game2048Screen)


@pytest.mark.asyncio
async def test_arrow_keys_move_tiles(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+g")
        await pilot.pause()
        screen = pilot.app.screen
        assert isinstance(screen, Game2048Screen)

        before = [row[:] for row in screen.game.grid]
        # Try all four directions; at least one must change the board.
        changed = False
        for key in ("up", "down", "left", "right"):
            await pilot.press(key)
            await pilot.pause()
            if screen.game.grid != before:
                changed = True
                break
        assert changed

        # Restart yields a fresh board with exactly two tiles.
        await pilot.press("r")
        await pilot.pause()
        tiles = [v for row in screen.game.grid for v in row if v]
        assert len(tiles) == 2
        assert screen.game.score == 0


@pytest.mark.asyncio
async def test_game_does_not_interrupt_running_agent(tmp_path):
    """The agent turn continues while the game modal is open."""
    gate = asyncio.Event()
    fake_llm = FakeLLMClient([[TextChunk(text="agent finished")]], gate=gate)
    app = make_app(tmp_path, fake_llm)
    async with app.run_test() as pilot:
        await pilot.pause()
        await submit(pilot, "do work")
        for _ in range(100):
            await pilot.pause()
            if fake_llm.calls == 1:
                break
        assert fake_llm.calls == 1

        # Open the game mid-turn (input is disabled, ctrl+g still works),
        # play a move, then close it.
        await pilot.press("ctrl+g")
        await pilot.pause()
        assert isinstance(pilot.app.screen, Game2048Screen)
        await pilot.press("left")
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(pilot.app.screen, MainScreen)

        # Let the agent finish; its reply must still arrive.
        gate.set()
        chat = pilot.app.screen.query_one("#chat", ChatWidget)
        for _ in range(200):
            await pilot.pause()
            if "agent finished" in chat.transcript_text():
                return
        raise AssertionError("agent turn did not complete after game closed")
