"""Tests for the XTermParser key fixes (herdr Shift+Enter)."""

from __future__ import annotations

from textual._xterm_parser import XTermParser

from limbo.ui.key_fixes import apply_key_fixes

apply_key_fixes()


def _keys_for(sequence: str) -> list[str]:
    parser = XTermParser()
    return [event.key for event in parser._sequence_to_key_events(sequence)]


def test_modify_other_keys_shift_enter():
    """herdr forwards Shift+Enter as ESC [ 27 ; 2 ; 13 ~ (modifyOtherKeys)."""
    assert _keys_for("\x1b[27;2;13~") == ["shift+enter"]


def test_kitty_shift_enter():
    """The standard kitty encoding maps to shift+enter even when Textual's
    kitty parsing is disabled (TEXTUAL_DISABLE_KITTY_KEY)."""
    assert _keys_for("\x1b[13;2u") == ["shift+enter"]


def test_other_sequences_unaffected():
    assert _keys_for("a") == ["a"]
    assert _keys_for("\x1b[A") == ["up"]
    assert _keys_for("\r") == ["enter"]


def test_apply_key_fixes_is_idempotent():
    patched = XTermParser._sequence_to_key_events
    apply_key_fixes()
    apply_key_fixes()
    assert XTermParser._sequence_to_key_events is patched
