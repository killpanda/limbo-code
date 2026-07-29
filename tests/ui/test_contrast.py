"""Palette contrast invariants (RFC LIM-16 v1.1 P2-3 / Test Plan §7.5).

The full foreground × background matrix lives in ``limbo.ui.contrast`` and
can be inspected via ``scripts/check_contrast.py``. These tests pin the
acceptance rule: every *used* text/background pair ≥ 4.5:1, nothing < 3.0:1.
"""

from __future__ import annotations

from limbo.ui.contrast import USED_PAIRS, check_palettes, check_theme
from limbo.ui.theme import LIMBO_DARK, LIMBO_LIGHT, palette_snapshot


def test_all_used_pairs_meet_text_threshold():
    results, failures = check_palettes()
    assert failures == 0, [
        f"{r.theme}: {r.fg} on {r.bg} = {r.ratio:.2f} ({r.status})"
        for r in results
        if r.status in ("FAIL", "BAD")
    ]


def test_matrix_enumerates_all_foregrounds_over_three_backgrounds():
    """The matrix must cover 前景阶 × 三级背景, including boundary combos like
    text-tertiary on bg-elevated (4.11:1 in limbo-dark)."""
    for theme in (LIMBO_DARK, LIMBO_LIGHT):
        pairs = {(r.fg, r.bg) for r in check_theme(theme.name, palette_snapshot(theme))}
        for fg in ("text-primary", "text-secondary", "text-tertiary"):
            for bg in ("bg-base", "bg-surface", "bg-elevated"):
                assert (fg, bg) in pairs
        # Boundary combination flagged by review must be reported.
        boundary = next(
            r for r in check_theme(theme.name, palette_snapshot(theme))
            if (r.fg, r.bg) == ("text-tertiary", "bg-elevated")
        )
        assert boundary.status in ("OK", "WARN")  # reported, never silently dropped


def test_used_pairs_cover_error_block_and_selection():
    assert ("error-bright", "error-bg") in USED_PAIRS
    assert ("accent", "bg-elevated") in USED_PAIRS
