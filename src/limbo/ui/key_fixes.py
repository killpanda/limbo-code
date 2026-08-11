"""Compatibility fixes for terminal key escape sequences Textual mis-handles.

Inside the herdr multiplexer, Limbo disables Textual's kitty keyboard
protocol (``ui.kitty_keyboard``, see ``KittyKeyboardMode``) so IME input
works. herdr still encodes Shift+Enter for the pane, using the xterm
*modifyOtherKeys* form ``ESC [ 27 ; 2 ; 13 ~``. Textual's legacy parser
(kitty disabled) does not know that sequence at all: the ESC is delivered
as a bare Escape keypress and the rest is reissued as literal text, so the
user sees ``[27;2;13~`` typed into the input instead of a newline.

Even with the kitty protocol enabled, Textual 8.2.8 mis-parses that
sequence as ``shift+\r`` (the standard kitty encoding of Shift+Enter is
``ESC [ 13 ; 2 u``, which parses correctly). Mapping the modifyOtherKeys
form explicitly is therefore strictly better in every mode.
"""

from __future__ import annotations

from textual import events
from textual._xterm_parser import XTermParser

# Escape sequences that must reach the app as a `shift+enter` key event:
# - "\x1b[27;2;13~": xterm modifyOtherKeys encoding (what herdr forwards)
# - "\x1b[13;2u":    standard kitty encoding (already parsed correctly when
#   the protocol is enabled, but not when TEXTUAL_DISABLE_KITTY_KEY is set)
_SHIFT_ENTER_SEQUENCES = frozenset({"\x1b[27;2;13~", "\x1b[13;2u"})

_original_sequence_to_key_events = XTermParser._sequence_to_key_events


def _sequence_to_key_events(self, sequence: str, alt: bool = False):
    if sequence in _SHIFT_ENTER_SEQUENCES:
        yield events.Key("shift+enter", None)
        return
    yield from _original_sequence_to_key_events(self, sequence, alt)


def apply_key_fixes() -> None:
    """Install the XTermParser shim (idempotent)."""
    if XTermParser._sequence_to_key_events is not _sequence_to_key_events:
        XTermParser._sequence_to_key_events = _sequence_to_key_events  # type: ignore[method-assign]
