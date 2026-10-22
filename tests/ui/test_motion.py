"""Limbo Motion: mode resolution, the shared clock and pure frame functions."""

from __future__ import annotations

from limbo.ui import motion

STOPS = ("#3069A9", "#5B9BF0", "#FDE75C")


def test_resolve_mode_precedence():
    assert motion.resolve_mode(None, {}) == motion.FULL
    assert motion.resolve_mode("reduced", {}) == motion.REDUCED
    assert motion.resolve_mode("bogus", {}) == motion.FULL
    assert motion.resolve_mode("full", {"LIMBO_MOTION": "off"}) == motion.OFF
    assert motion.resolve_mode("full", {"NO_COLOR": "1"}) == motion.OFF
    assert motion.resolve_mode(None, {"TERM": "dumb"}) == motion.OFF
    assert motion.resolve_mode(None, {"SSH_TTY": "/dev/pts/1"}) == motion.REDUCED
    assert motion.resolve_mode("full", {"SSH_TTY": "/dev/pts/1"}) == motion.FULL


class FakeTimer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeApp:
    def __init__(self) -> None:
        self.timers: list[FakeTimer] = []

    def set_interval(self, _interval, _callback):
        timer = FakeTimer()
        self.timers.append(timer)
        return timer


def test_clock_runs_only_while_subscribed():
    app = FakeApp()
    clock = motion.MotionClock(app, motion.FULL)
    seen: list[int] = []
    owner = object()
    clock.subscribe(owner, seen.append)
    clock.subscribe(object(), lambda _s: 1 / 0)  # broken subscriber is isolated
    assert len(app.timers) == 1
    clock._tick()
    clock._tick()
    assert seen == [1, 2]
    clock.unsubscribe(owner)
    assert not app.timers[0].stopped
    clock._subscribers.clear()
    clock.unsubscribe(owner)
    assert app.timers[0].stopped


def test_off_clock_never_starts_a_timer():
    app = FakeApp()
    clock = motion.MotionClock(app, motion.OFF)
    clock.subscribe(object(), lambda _s: None)
    assert app.timers == []


def test_tide_frame_is_deterministic_and_fixed_width():
    a = motion.tide_frame(7, STOPS, "#333333", seed=3)
    b = motion.tide_frame(7, STOPS, "#333333", seed=3)
    assert a.plain == b.plain
    assert len(a.plain) == motion.TIDE_WIDTH
    still = motion.tide_frame(7, STOPS, "#333333", mode=motion.REDUCED)
    assert still.plain == "╱" * motion.TIDE_WIDTH
    calm = motion.tide_frame(7, STOPS, "#333333", scramble=False)
    assert set(calm.plain) <= {"╱", "▓", "▒"}
    newborn = motion.tide_frame(0, STOPS, "#333333", birth=0)
    assert newborn.plain.strip() == ""


def test_decode_settles_left_to_right():
    final = "Read src/app.py"
    first = motion.decode_text(final, 1, "#444444", seed=1)
    assert first.plain.startswith(final[: motion.DECODE_PER_FRAME])
    assert len(first.plain) == len(final)
    done = motion.decode_text(final, motion.decode_frames(final), "#444444")
    assert done.plain == final
    assert motion.decode_text(final, 0, "#444444", mode=motion.OFF).plain == final
    assert motion.decode_frames("x" * 100) == 8


def test_settle_fades_out():
    assert motion.settle_color(0, "#000000", "#FFFFFF") != "#000000"
    assert motion.settle_color(motion.SETTLE_FRAMES + 1, "#000000", "#FFFFFF") is None
    assert motion.settle_color(0, "#000000", "#FFFFFF", motion.REDUCED) is None


def test_sweep_keeps_glyphs_and_ends_on_gradient():
    lines = ("AB C", "DEF ")
    frame0 = motion.sweep_lines(lines, STOPS, "#222222", 0)
    final = motion.sweep_lines(lines, STOPS, "#222222", 24)
    off = motion.sweep_lines(lines, STOPS, "#222222", 0, mode=motion.OFF)
    assert frame0.plain == final.plain == "AB C\nDEF "
    assert [s.style for s in final.spans] == [s.style for s in off.spans]


def test_spinner_and_ellipsis_respect_mode():
    assert motion.spinner(0) in motion.SPINNER
    assert motion.spinner(5, motion.REDUCED) == "●"
    assert motion.ellipsis(3, motion.OFF) == "…"
    assert motion.ellipsis(8) == "."


def test_get_clock_falls_back_to_off():
    class Bare:
        pass

    app = Bare()
    clock = motion.get_clock(app)
    assert clock.mode == motion.OFF
    assert motion.get_clock(app) is clock
