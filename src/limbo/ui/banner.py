"""Startup ASCII-art banner shown on the main screen.

Compact 7-row version downsampled from the original 33-row cartoon art
(red shirt / yellow pants on blue). Characters keep the original palette
colors; spaces carry **no style** so the banner blends into the active
theme background instead of painting a hardcoded blue rectangle (RFC
LIM-16 v1.1 P0-5).

Palette sampled from the source image (RGB):

- ``@`` dark outline #41415E
- ``o`` red shirt #ED5853
- ``.`` yellow pants #FDE75C
"""

from rich.text import Text

BLUE = (48, 105, 169)  # original background; kept for reference/tests
YELLOW = (253, 231, 92)  # pants
RED = (237, 88, 83)  # shirt
DARK = (65, 64, 94)  # outline

_FG = {"@": DARK, "o": RED, ".": YELLOW}

STARTUP_ART = """\
   @oo@@oooooooooooooo@ooooo@
   @@o@ooooooooooooooo@ooooo@
   o@@@@.@@@@@@@oooooo@@@@@@@
 oo oo@@..@@@@@.........@@@@@@
ooo@@ @@@@@..@@@........@@ooo
   o    o@.....@........@@
         @.....@........@"""


def startup_art_text() -> Text:
    """STARTUP_ART as a Rich Text colored with the original palette.

    Space characters are appended unstyled (transparent), so the art sits
    on the theme background rather than a solid blue block.
    """
    text = Text()
    for i, line in enumerate(STARTUP_ART.splitlines()):
        if i:
            text.append("\n")
        for ch in line:
            color = _FG.get(ch)
            if color is None:
                text.append(ch)
            else:
                text.append(ch, style=f"rgb({color[0]},{color[1]},{color[2]})")
    return text
