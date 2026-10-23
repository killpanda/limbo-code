"""Compatibility fixes for terminal key escape sequences Textual mis-handles.

Inside the herdr multiplexer, Limbo requests only the kitty *disambiguate*
flag (``ui.kitty_keyboard``, see ``KittyKeyboardMode``) so IME input works
and Shift+Enter still arrives as ``ESC [ 13 ; 2 u``. A pane that has asked
for xterm *modifyOtherKeys* gets Shift+Enter as ``ESC [ 27 ; 2 ; 13 ~``
instead. Textual's legacy parser (kitty disabled) does not know that
sequence at all: the ESC is delivered as a bare Escape keypress and the rest
is reissued as literal text, so the user sees ``[27;2;13~`` typed into the
input instead of a newline.

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
# Same story for Alt+Enter (queue a follow-up): modifyOtherKeys modifier 3
# is parsed as a bare "\r", and the legacy ESC-prefixed form loses the Alt.
_ALT_ENTER_SEQUENCES = frozenset({"\x1b[27;3;13~", "\x1b[13;3u", "\x1b\r"})

_original_sequence_to_key_events = XTermParser._sequence_to_key_events


def _sequence_to_key_events(self, sequence: str, alt: bool = False):
    if sequence in _SHIFT_ENTER_SEQUENCES:
        yield events.Key("shift+enter", None)
        return
    if sequence in _ALT_ENTER_SEQUENCES:
        yield events.Key("alt+enter", None)
        return
    yield from _original_sequence_to_key_events(self, sequence, alt)


def apply_key_fixes() -> None:
    """Install the XTermParser shim (idempotent)."""
    if XTermParser._sequence_to_key_events is not _sequence_to_key_events:
        XTermParser._sequence_to_key_events = _sequence_to_key_events  # type: ignore[method-assign]


def request_kitty_disambiguate_only() -> None:
    """Make Textual request only the kitty ``DISAMBIGUATE_ESCAPE_CODES`` flag.

    Textual hardcodes ``DISAMBIGUATE | REPORT_ALL_KEYS | REPORT_ASSOCIATED_TEXT``
    in ``LinuxDriver.start_application_mode``, reading the flag constants as
    module globals at call time; zeroing the two extra flags leaves ``>1u``.
    """
    try:
        from textual.drivers import linux_driver  # noqa: PLC0415 - POSIX only
    except ImportError:  # Windows: no termios, no kitty request either
        return
    linux_driver.KITTY_REPORT_ALL_KEYS = 0  # type: ignore[misc]
    linux_driver.KITTY_REPORT_ASSOCIATED_TEXT = 0  # type: ignore[misc]
