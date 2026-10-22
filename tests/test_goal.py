"""Tests for the goal-mode pure functions and the verify executor (LIM-40)."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.goal import (
    STATUS_ACTIVE,
    GoalState,
    VerifyResult,
    next_state,
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    return tmp_path


# -- parse_goal_args ----------------------------------------------------------


# -- M2: proposer prompt + proposal parsing ------------------------------------


# -- prompt templates ---------------------------------------------------------


# -- next_state ---------------------------------------------------------------


def test_next_state_failure_stays_active_within_budget():
    state = GoalState(text="g", max_rounds=10)
    new = next_state(state, VerifyResult(exit_code=1))
    assert new.status == STATUS_ACTIVE
    assert new.rounds_completed == 1


# -- run_verify ---------------------------------------------------------------


