"""Cheap updates for one-line widgets (long-session lag).

``Static.update`` defaults to ``layout=True``, and Textual answers any layout
request by re-arranging every widget on the screen. Spinners, the activity
line and running tool-card headers update 8-20 times a second, so in a long
conversation they kept the event loop busy with full-screen reflows even
though their size never changed. ``update_line`` only asks for a layout
when the line's footprint can actually change.
"""

from __future__ import annotations

from rich.cells import cell_len
from rich.text import Text
from textual.widgets import Static


def update_line(widget: Static, content: str | Text) -> None:
    """``widget.update(content)``, skipping the reflow when the size is stable.

    A single line keeps its size when its cell width is unchanged, or — for a
    widget whose width does not follow its content — when both the old and
    the new line fit the current width (no wrap, so the height stays 1).
    """
    plain = content.plain if isinstance(content, Text) else content
    cells = None if "\n" in plain else cell_len(plain)
    previous = getattr(widget, "_limbo_line_cells", None)
    widget._limbo_line_cells = cells  # type: ignore[attr-defined]
    if cells is None or previous is None:
        layout = True
    elif cells == previous:
        layout = False
    elif widget.styles.width is not None and widget.styles.width.is_auto:
        layout = True
    else:
        width = widget.content_size.width
        layout = not (width > 0 and previous <= width and cells <= width)
    widget.update(content, layout=layout)
