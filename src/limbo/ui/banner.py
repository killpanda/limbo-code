"""Startup ASCII-art banner shown on the main screen.

Generated from the project's cartoon image (red shirt / yellow pants on a
blue background) by quantizing each pixel to the nearest of four palette
colors and mapping them to characters:

- ``@`` dark outline (line art preserved with a per-cell coverage threshold)
- ``o`` red shirt
- ``.`` yellow pants
- `` `` (space) blue background

``startup_art_text()`` renders the same characters as a Rich ``Text`` with
the original image colors: each cell's foreground is its palette color on
the blue background, so the banner looks like the source cartoon.
"""

from rich.text import Text

# Palette sampled from the source image (RGB).
BLUE = (48, 105, 169)  # background
YELLOW = (253, 231, 92)  # pants
RED = (237, 88, 83)  # shirt
DARK = (65, 64, 94)  # outline

_FG = {"@": DARK, "o": RED, ".": YELLOW, " ": BLUE}


def _style(char: str) -> str:
    r, g, b = _FG[char]
    bg = BLUE
    return f"rgb({r},{g},{b}) on rgb({bg[0]},{bg[1]},{bg[2]})"


def startup_art_text() -> Text:
    """STARTUP_ART as a Rich Text colored with the original image palette.

    Lines are padded to a uniform width so the blue background forms the
    same solid rectangle as the source image.
    """
    lines = STARTUP_ART.splitlines()
    width = max(len(line) for line in lines)
    text = Text()
    for i, line in enumerate(lines):
        if i:
            text.append("\n")
        for ch in line.ljust(width):
            text.append(ch, style=_style(ch))
    return text


STARTUP_ART = """\
      @@ooooo@@ooooooooooooooooooooooooooooo@@oooooooooo@@
       @ooooo@@ooooooooooooooooooooooooooooo@@oooooooooo@@
       @@oooo@@ooooooooooooooooooooooooooooo@@oooooooooo@@
       @@oooo@oooooooooooooooooooooooooooooo@@oooooooooo@@
       @@ooo@@oooooooooooooooooooooooooooooo@@oooooooooo@@
       @@ooo@@oooooooooooooooooooooooooooooo@@oooooooooo@@
       @@ooo@@oooooooooooooooooooooooooooooo@@oooooooooo@@
       @@ooo@@oooooooooooooooooooooooooooooo@@ooooooooooo@
       @@ooo@@oooooooooooooooooooooooooooooo@@ooooooooooo@
       @@ooo@@@ooooooooooooooooooooooooooooo@@oooooooooo@@
        @@ooo@@@@@@@@ooooooooooooooooooooooo@@@ooooooooo@@
         @@@@@@....@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ooooooo@@
          @@@@@................................@@@@@@@@@@
            @@@.......@@@@.@@@@................@@@@@@@@@@@
            @@@.......@@@@..@...................@@@@@@...@
             @@........@@@..@...................@@......@@@
              @@.......@@@......................@@..@.@@@@@
         @@@@ @@...@@..@@@@.....................@@@@@@@@@@@
  @@@          @@..@@@....@@@@...................@@@@@@@@@
 @@     @     @@@@...@...@@@@....................@@@@@@@@
     @@@@@      @@.@@@....@@@....................@@
             @@  @@.........@@@@.................@@
            @@   @@...........@@.................@@
      @@         @@...........@@.................@@
      @@         @@...........@@.................@@
                 @@...........@@.................@@
                  @...........@@.................@@
                  @...........@@.................@@
                  @@..........@@.................@
                  @@...........@................@@
                  @@...........@................@@
                  @@...........@@...............@@"""
