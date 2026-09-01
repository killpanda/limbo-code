"""Startup banner shown on fresh sessions.

Half-block truecolor rendering (RFC LIM-16 v1.1 P0-5 successor):
``STARTUP_ART`` is a pixel map classified from the 640x640 mascot image
by ``scripts/gen_banner.py`` (BOX downscale -> nearest-palette color ->
despeckle -> crop). Each pair of pixel rows renders as one terminal row
using half-block glyphs, doubling vertical resolution versus
one-character-per-pixel art and keeping anti-aliased edges via
foreground/background color mixing.

Palette keys (colors sampled from the source image):

- ``D`` outline
- ``R`` red shirt
- ``Y`` yellow pants
- space: transparent — carries **no style**, so the banner blends into
  the active theme background instead of painting a hardcoded blue
  rectangle.

Regenerate after replacing the mascot:

    python scripts/gen_banner.py /path/to/image.jpg --width 40
"""

from rich.text import Text

BLUE = (48, 105, 169)  # original background; transparent at render time
YELLOW = (253, 231, 92)  # pants
RED = (237, 88, 83)  # shirt
DARK = (65, 64, 94)  # outline (brightened vs. source for dark themes)

_PALETTE = {"R": RED, "Y": YELLOW, "D": DARK}

UPPER = "▀"  # top pixel as foreground
LOWER = "▄"  # bottom pixel as foreground
FULL = "█"  # both pixels share one color

STARTUP_ART = """\
DRRRDRRRRRRRRRRRRRRRRRRRRRRRRRRD
DRRRDRRRRRRRRRRRRRRRRRRRRRRRRRRD
DRRRDRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRDRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRDRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRD
 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
 RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR
 DRRDRRRRRRRRRRRRRRRRRRRRRRRRRRR
 DRRDDRRRRRRRRRRRRRRRRRRDRRRRRRD
 DRRDDRDDRRRRRRRRRRRRRRRDRRRRRRD
  DRDRYYYYRRRDDDDDDDDDRYYRRRRRRD
   DDYYYYYYYYYYYYYYYYYYYYDRRRDDD
   D YYYYYDYYYYYYYYYYYYYYYDRRRDD
    DYYYYYYDYYYYYYYYYYYYYYDDDDR
    DRYYYYYDYYYYYYYYYYYYYYYYYYYR
    DRYYYYYDYYYYYYYYYYYYYYDYYYYRD
     DYYYYRYYYYYYYYYYYYYYYDYYRR D
     DYYYYRDYYYYYYYYYYYYYYRRRR DD
     DYYYRYYDDYYYYYYYYYYYYRRRRDD
     DDYYYYYYRRYYYYYYYYYYYR  RDD
     DDYYYYYDYYYYYYYYYYYYYYDDDD
      DYRYYYDYYYYYYYYYYYYYYD
       RYYYYYYDYYYYYYYYYYYYD
       RYYYYYYYDYYYYYYYYYYYD
       DYYYYYYYDYYYYYYYYYYYD
       DYYYYYYYDYYYYYYYYYYYD
       DYYYYYYYRYYYYYYYYYYYD
       DYYYYYYYRYYYYYYYYYYYD
       DYYYYYYYDYYYYYYYYYYYD
       DYYYYYYYDYYYYYYYYYYY
       DYYYYYYYDYYYYYYYYYYR
       DYYYYYYYDYYYYYYYYYYR
       DYYYYYYYDYYYYYYYYYYR
       DYYYYYYYDYYYYYYYYYYD
       DYYYYYYYRYYYYYYYYYYD
       DYYYYYYYRYYYYYYYYYYD"""


def _rgb(c: tuple[int, int, int]) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def startup_art_text() -> Text:
    """STARTUP_ART pixel map as a Rich Text of half-block cells.

    Each pair of pixel rows becomes one terminal row: ``▀`` (top pixel as
    foreground), ``▄`` (bottom pixel as foreground), ``█`` (both pixels
    share a color), or ``▀`` with a foreground/background style when the
    two pixels differ. Fully transparent pairs stay unstyled spaces, so
    the art sits on the theme background rather than a solid blue block.
    """
    rows = STARTUP_ART.splitlines()
    if len(rows) % 2:
        rows.append("")
    width = max(len(r) for r in rows)
    text = Text()
    for i in range(0, len(rows), 2):
        if i:
            text.append("\n")
        top, bottom = rows[i], rows[i + 1]
        for x in range(width):
            ct = _PALETTE.get(top[x]) if x < len(top) else None
            cb = _PALETTE.get(bottom[x]) if x < len(bottom) else None
            if ct is None and cb is None:
                text.append(" ")
            elif ct is not None and cb is not None:
                if ct == cb:
                    text.append(FULL, style=_rgb(ct))
                else:
                    text.append(UPPER, style=f"{_rgb(ct)} on {_rgb(cb)}")
            elif ct is not None:
                text.append(UPPER, style=_rgb(ct))
            elif cb is not None:
                text.append(LOWER, style=_rgb(cb))
    return text


def user_icon() -> Text:
    """A compact single-row mascot icon for user messages (LIM-16 v1.1).

    Half-block compression of the mascot: red shirt on top, yellow pants
    below, dark outline on the sides. One terminal row tall so it lines up
    with the message text ("same size as the font") while keeping the
    brand colors. Prepended before the ``❯`` user-message marker.

    ``DRRRD`` / ``DYYYD`` -> 2 px rows condensed into 1 terminal row via
    ``▀`` (top pixel as foreground, bottom pixel as background).
    """
    top = "DRRRD"
    bottom = "DYYYD"
    text = Text()
    for ct_key, cb_key in zip(top, bottom):
        ct = _PALETTE[ct_key]
        cb = _PALETTE[cb_key]
        if ct == cb:
            text.append(FULL, style=_rgb(ct))
        else:
            text.append(UPPER, style=f"{_rgb(ct)} on {_rgb(cb)}")
    return text
