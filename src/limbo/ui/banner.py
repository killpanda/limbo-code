"""Startup ASCII-art banner shown on the main screen.

Generated from the project's cartoon image (red shirt / yellow pants on a
blue background) by quantizing each pixel to the nearest of four palette
colors and mapping them to characters:

- ``@`` dark outline (line art preserved with a per-cell coverage threshold)
- ``o`` red shirt
- ``.`` yellow pants
- `` `` (space) blue background
"""

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
