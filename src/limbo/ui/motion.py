"""Limbo Motion: one 20 fps clock and the frame functions it drives.

Design (``limbo-motion`` spec): motion tells the story "from unsettled to
settled". Every animation hangs off a single :class:`MotionClock` that only
runs while something subscribes, so an idle limbo costs no CPU. Frame
functions are pure — ``(step, ...) -> Text`` with seeded randomness — so a
snapshot test that fixes the step gets byte-identical output.

Effects:

- **Tide** (thinking / running): a row of still ``╱`` with a bright crest
  sweeping through; only cells near the crest scramble. ``scramble=False``
  is the tool variant (light only, no glyphs).
- **Decode**: text resolves left to right out of dim scramble.
- **Settle**: one-shot background flash that fades over 6 frames.
- **Sweep**: a band of light lights up the wordmark once.

``[ui] motion = "full" | "reduced" | "off"``. ``reduced`` drops scramble,
flashes and drift but keeps state changes; ``off`` renders final frames.
``LIMBO_MOTION`` overrides the config (the test suite pins ``off``);
``NO_COLOR`` / ``TERM=dumb`` force ``off`` and SSH sessions default to
``reduced``.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Mapping
from typing import Any

from rich.style import Style
from rich.text import Text

from limbo.ui.gradient import mix, ramp

FPS = 20
FULL, REDUCED, OFF = "full", "reduced", "off"
MODES = (FULL, REDUCED, OFF)

TIDE_WIDTH = 14
TIDE_GLYPHS = "╱╲░▒▓◇∴·※"
SCRAMBLE = "abcdefghijkmnpqrstuvwxyz0123456789#%&*+=?/<>"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
ELLIPSIS = ("", ".", "..", "...")
SETTLE_FRAMES = 6
DECODE_PER_FRAME = 3


def resolve_mode(configured: str | None, env: Mapping[str, str] | None = None) -> str:
    """Effective motion mode from config + environment."""
    env = os.environ if env is None else env
    override = env.get("LIMBO_MOTION", "").strip().lower()
    if override in MODES:
        return override
    if "NO_COLOR" in env or env.get("TERM") == "dumb":
        return OFF
    value = (configured or "").strip().lower()
    if value in MODES:
        return value
    # Full-row redraws at 20 fps stutter over high-latency links.
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return REDUCED
    return FULL


class MotionClock:
    """App-wide 20 fps tick; starts with the first subscriber, stops with the last."""

    def __init__(self, app: Any, mode: str = FULL) -> None:
        self._app = app
        self.mode = mode
        self.step = 0
        self._subscribers: dict[int, Callable[[int], None]] = {}
        self._timer: Any = None

    @property
    def animated(self) -> bool:
        return self.mode != OFF

    def subscribe(self, owner: object, callback: Callable[[int], None]) -> None:
        self._subscribers[id(owner)] = callback
        if self._timer is None and self.animated:
            self._timer = self._app.set_interval(1 / FPS, self._tick)

    def unsubscribe(self, owner: object) -> None:
        self._subscribers.pop(id(owner), None)
        if not self._subscribers and self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        self.step += 1
        for callback in list(self._subscribers.values()):
            try:
                callback(self.step)
            except Exception:  # noqa: BLE001 - one broken widget must not stop the clock
                continue


# -- frame functions (pure) -----------------------------------------------------


def tide_frame(
    step: int,
    stops: tuple[str, ...],
    dim: str,
    *,
    mode: str = FULL,
    scramble: bool = True,
    seed: int = 0,
    width: int = TIDE_WIDTH,
    birth: int | None = None,
) -> Text:
    """One frame of the Tide indicator.

    ``birth`` is the step the indicator appeared: for the first ~1s cells
    are born left to right (borrowed from crush's staggered entrance).
    """
    text = Text()
    span = max(1, width - 1)
    if mode != FULL:
        for i in range(width):
            text.append("╱", Style(color=ramp(stops, i / span)))
        return text
    rng = random.Random(seed * 100_003 + step)
    period = int((width + 8) / 0.6) + 6
    crest = (step % period) * 0.6 - 4
    born = width if birth is None else int((step - birth) / 1.4)
    for i in range(width):
        if i >= born:
            text.append(" ")
            continue
        base = ramp(stops, i / span)
        distance = abs(i - crest)
        if distance < 2.2:
            glyph = rng.choice(TIDE_GLYPHS) if scramble else ("▓" if distance < 1 else "▒")
            color = mix(base, "#FFFFFF", max(0.0, 0.35 - distance * 0.15))
            text.append(glyph, Style(color=color))
        else:
            near = max(0.0, 1 - (distance - 2.2) / 4)
            text.append("╱", Style(color=mix(dim, base, near * 0.7)))
    return text


def ellipsis(step: int, mode: str = FULL) -> str:
    if mode == OFF:
        return "…"
    return ELLIPSIS[(step // 8) % len(ELLIPSIS)]


def spinner(step: int, mode: str = FULL) -> str:
    if mode != FULL:
        return "●"
    return SPINNER[(step // 2) % len(SPINNER)]


def decode_text(final: str, frames: int, dim: str, *, mode: str = FULL, seed: int = 0) -> Text:
    """``final`` with the first ``frames * DECODE_PER_FRAME`` chars settled."""
    settled = len(final) if mode != FULL else frames * DECODE_PER_FRAME
    if settled >= len(final):
        return Text(final)
    rng = random.Random(seed * 7919 + frames)
    text = Text(final[:settled])
    rest = "".join(" " if ch.isspace() else rng.choice(SCRAMBLE) for ch in final[settled:])
    text.append(rest, Style(color=dim))
    return text


def decode_frames(final: str) -> int:
    """Frames a decode of ``final`` takes (capped at 8 = 400ms)."""
    return min(8, -(-len(final) // DECODE_PER_FRAME))


def settle_color(frame: int, base: str, flash: str, mode: str = FULL) -> str | None:
    """Background for frame ``frame`` of a Settle flash; None once faded."""
    if mode != FULL or frame > SETTLE_FRAMES:
        return None
    return mix(base, flash, 0.22 * (1 - frame / SETTLE_FRAMES))


def sweep_lines(
    lines: tuple[str, ...],
    stops: tuple[str, ...],
    dim: str,
    frame: int,
    *,
    total: int = 24,
    mode: str = FULL,
    reverse: bool = False,
) -> Text:
    """Wordmark lit by a band of light moving left to right (Dusk Sweep).

    ``reverse`` sweeps right to left (the daily mode divider on the way
    back to coding); colours stay anchored left to right either way.
    """
    width = max((len(line) for line in lines), default=1)
    span = max(1, width - 1)
    front = (min(frame, total) / total) * (width + 12) - 6 if mode == FULL else width + 12
    text = Text()
    for row, line in enumerate(lines):
        if row:
            text.append("\n")
        for col, ch in enumerate(line):
            if ch.isspace():
                text.append(ch)
                continue
            base = ramp(stops, col / span)
            pos = width - 1 - col if reverse else col
            if pos > front + 3:
                color = mix(dim, base, 0.15)
            elif pos > front - 3:
                color = mix(base, "#FFFFFF", 0.45 * (1 - abs(pos - front) / 3))
            else:
                color = base
            text.append(ch, Style(color=color, bold=True))
    return text


def get_clock(app: Any) -> MotionClock:
    """The app's clock; apps that never installed one get a static (off) clock."""
    clock = getattr(app, "motion", None)
    if isinstance(clock, MotionClock):
        return clock
    clock = MotionClock(app, OFF)
    try:
        app.motion = clock
    except Exception:  # noqa: BLE001 - read-only app stand-ins in tests
        pass
    return clock
