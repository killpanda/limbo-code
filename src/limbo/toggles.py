"""Argument parsing shared by on/off slash commands (/code-mode, /plan)."""

from __future__ import annotations

_ON = frozenset({"on", "1", "true", "yes", "开", "开启"})
_OFF = frozenset({"off", "0", "false", "no", "关", "关闭"})


def parse_toggle(arg: str, current: bool) -> bool | None:
    """``on``/``off`` → that state, empty → flip ``current``, else None."""
    arg = arg.strip().lower()
    if arg in _ON:
        return True
    if arg in _OFF:
        return False
    if arg == "":
        return not current
    return None
