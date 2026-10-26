"""Assistant replies as one widget each (long-session lag).

Textual's ``Markdown`` widget mounts one child widget per block (paragraph,
bullet, fence, ...) — a typical reply is 50-100 widgets. Any layout refresh
anywhere on the screen (a spinner tick, a new tool card, a streamed chunk)
re-arranges every widget in the DOM, so a long conversation made every frame
slower: measured ~130 ms per reflow after only three tool-heavy turns. Each
streamed append also built and threw away block widgets, feeding the
garbage collector.

``MarkdownView`` renders the source with Rich's Markdown — styled with the
same theme tokens as the ``Markdown*`` rules in ``app.tcss`` — into
``Content`` (so drag-select copy keeps working), once per (width, theme).
While streaming, completed blocks (up to the last blank line outside a code
fence) are rendered once and kept; only the unfinished tail is re-rendered
per flush. ``finish`` re-renders the whole reply in one piece, matching
how restored history renders.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

from rich._loop import loop_first
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Heading, HorizontalRule, ListItem, Markdown
from rich.rule import Rule
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme as RichTheme
from textual.content import Content
from textual.geometry import Size
from textual.selection import Selection
from textual.widget import Widget

from limbo.ui.syntax import style_for_theme

_FALLBACK_WIDTH = 80


class _Heading(Heading):
    """Left-aligned headings (Rich centers h1); h1 gets a bottom rule."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        yield text
        if self.tag == "h1":
            yield Text("─" * options.max_width, style="markdown.h1.border")


# Highlighting is the expensive, width-independent part of rendering a
# reply; re-renders on resize only re-wrap. Keyed by (code, lexer, style).
_HIGHLIGHT_CACHE: OrderedDict[tuple[str, str, Any], Text] = OrderedDict()
_HIGHLIGHT_CACHE_SIZE = 512


def _highlight(code: str, lexer: str, theme: Any) -> Text:
    key = (code, lexer, theme)
    text = _HIGHLIGHT_CACHE.get(key)
    if text is None:
        text = Syntax(code, lexer, theme=theme).highlight(code)
        text.rstrip()
        _HIGHLIGHT_CACHE[key] = text
        if len(_HIGHLIGHT_CACHE) > _HIGHLIGHT_CACHE_SIZE:
            _HIGHLIGHT_CACHE.popitem(last=False)
    else:
        _HIGHLIGHT_CACHE.move_to_end(key)
    return text


class _Fence(CodeBlock):
    """Code block on the panel background with a thick primary left bar."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        code = str(self.text).rstrip()
        bar = Segment("▌", console.get_style("markdown.fence.border"))
        background = console.get_style("markdown.code_block")
        # bar + 1 cell padding on each side.
        width = max(1, options.max_width - 3)
        pad = Segment(" ", background)
        text = _highlight(code, self.lexer_name, self.theme)
        for line in text.split("\n", allow_blank=True):
            for wrapped in line.wrap(console, width, overflow="fold", no_wrap=False):
                wrapped.stylize(background)
                wrapped.pad_right(width - wrapped.cell_len)
                yield bar
                yield pad
                yield from wrapped.render(console)
                yield pad
                yield Segment.line()


class _ListItem(ListItem):
    """Ordered items as ``1.`` (Rich prints a bare ``1``)."""

    def render_number(
        self, console: Console, options: ConsoleOptions, number: int, last_number: int
    ) -> RenderResult:
        width = len(str(last_number)) + 3
        lines = console.render_lines(
            self.elements, options.update(width=options.max_width - width), style=self.style
        )
        style = console.get_style("markdown.item.number", default="none")
        numeral = Segment(f"{number}.".rjust(width - 1) + " ", style)
        padding = Segment(" " * width, style)
        for first, line in loop_first(lines):
            yield numeral if first else padding
            yield from line
            yield Segment.line()


class _Rule(HorizontalRule):
    """A solid rule, like the MarkdownHorizontalRule widget."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield Rule(style=console.get_style("markdown.hr", default="none"))
        yield Text()


class _LimboMarkdown(Markdown):
    elements = {
        **Markdown.elements,
        "heading_open": _Heading,
        "fence": _Fence,
        "code_block": _Fence,
        "list_item_open": _ListItem,
        "hr": _Rule,
    }


def _rich_theme(variables: Mapping[str, str]) -> RichTheme:
    """Map the app's CSS variables to Rich markdown styles (mirrors app.tcss)."""
    v = variables
    fg = v.get("foreground", "#e0e0e0")
    primary = v.get("primary", fg)
    accent = v.get("accent", primary)
    panel = v.get("panel", v.get("background", "#000000"))
    tertiary = v.get("text-tertiary", fg)
    secondary = v.get("text-secondary", fg)
    border = v.get("border-subtle", tertiary)
    heading = Style(color=accent, bold=True)
    return RichTheme(
        {
            "markdown.h1": heading,
            "markdown.h2": heading,
            "markdown.h3": heading,
            "markdown.h4": heading,
            "markdown.h5": heading,
            "markdown.h6": heading,
            "markdown.h1.border": Style(color=border),
            "markdown.code": Style(color=accent, bgcolor=panel),
            "markdown.code_block": Style(bgcolor=panel),
            "markdown.fence.border": Style(color=primary, bgcolor=panel),
            "markdown.block_quote": Style(color=tertiary),
            "markdown.hr": Style(color=tertiary),
            "markdown.item.bullet": Style(color=secondary),
            "markdown.item.number": Style(color=secondary),
            "markdown.list": Style(),
            "markdown.link": Style(color=primary),
            "markdown.link_url": Style(color=primary, underline=True),
            "markdown.table.border": Style(color=border),
            "markdown.table.header": Style(color=accent, bold=True),
        }
    )


def render_markdown(
    source: str, width: int, theme: Any, variables: Mapping[str, str]
) -> Text:
    """Render ``source`` to a Text of pre-wrapped lines at ``width`` cells.

    ``theme`` picks the code highlighting; ``variables`` (the app's resolved
    CSS variables) color everything else.
    """
    console = Console(
        width=width,
        theme=_rich_theme(variables),
        color_system="truecolor",
        force_terminal=True,
        legacy_windows=False,
    )
    # Rich types code_theme as a theme name; _Fence hands it to Syntax,
    # which also takes a Pygments style class (same as the tool cards).
    md = _LimboMarkdown(source, code_theme=style_for_theme(theme))  # type: ignore[arg-type]
    lines = console.render_lines(md, console.options, pad=False)
    # Rich ends blocks with a blank separator line; drop trailing blanks so
    # the widget's height matches the live Markdown widget it replaces.
    while lines and not "".join(seg.text for seg in lines[-1]).strip():
        lines.pop()
    text = Text(end="")
    for i, line in enumerate(lines):
        if i:
            text.append("\n")
        for segment in line:
            if not segment.control:
                text.append(segment.text, segment.style)
    return text


def stable_end(source: str) -> int:
    """Offset just past the last blank line outside a code fence.

    Everything before it is complete Markdown blocks that later text can no
    longer change (lazy continuation lines aside, which only matter while
    streaming — ``finish`` re-renders the whole reply).
    """
    end = 0
    pos = 0
    fence: str | None = None
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
        elif stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
        elif not stripped and line.endswith("\n"):
            end = pos + len(line)
        pos += len(line)
    return end


_BLOCK_GAP = Content("\n\n")


class MarkdownView(Widget):
    """An assistant reply rendered as one widget (see module doc)."""

    DEFAULT_CSS = """
    MarkdownView {
        height: auto;
    }
    """

    def __init__(self, source: str = "", *, streaming: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._source = source
        self._streaming = streaming
        self._width = 0
        # Finished: rendered lines per (width, theme); two entries cover
        # toggling the sidebar back and forth without re-rendering.
        self._cache: OrderedDict[tuple[int, str], Content] = OrderedDict()
        # Streaming: completed blocks rendered once, for ``_chunks_key``.
        self._chunks: list[Content] = []
        self._chunked = 0
        self._chunks_key: tuple[int, str] | None = None

    @property
    def source(self) -> str:
        return self._source

    @property
    def streaming(self) -> bool:
        return self._streaming

    def append(self, text: str) -> None:
        """Add a streamed chunk (re-renders only the unfinished tail)."""
        if not text:
            return
        self._source += text
        self.refresh(layout=True)

    def finish(self) -> None:
        """The reply is complete: render it whole from now on."""
        if not self._streaming:
            return
        self._streaming = False
        self._chunks = []
        self._chunked = 0
        self._chunks_key = None
        self.refresh(layout=True)

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, self._on_theme_changed)

    def on_unmount(self) -> None:
        # Pruned from the chat (kept for paging): drop the rendered lines.
        self._cache.clear()

    def _on_theme_changed(self, _theme: Any) -> None:
        self._cache.clear()
        self._chunks_key = None
        self.refresh(layout=True)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        # Layout hands us the real width before rendering; the visual Textual
        # caches per layout pass must be rebuilt when the width changes
        # (terminal resize, sidebar toggle), or the pre-wrapped lines would
        # be cut / under-filled.
        if width and width != self._width:
            self._width = width
            self._layout_cache.pop("_render.visual", None)
        return super().get_content_height(container, viewport, width)

    def _render_source(self, source: str, width: int) -> Content:
        theme = self.app.current_theme
        return Content.from_rich_text(
            render_markdown(source, width, theme, self.app.theme_variables)
        )

    def render(self) -> Content:
        width = self._width or self.content_size.width or _FALLBACK_WIDTH
        key = (width, self.app.current_theme.name)
        if self._streaming:
            return self._render_streaming(key)
        content = self._cache.get(key)
        if content is None:
            content = self._render_source(self._source, width)
            self._cache[key] = content
            if len(self._cache) > 2:
                self._cache.popitem(last=False)
        return content

    def _render_streaming(self, key: tuple[int, str]) -> Content:
        width = key[0]
        if self._chunks_key != key:
            self._chunks = []
            self._chunked = 0
            self._chunks_key = key
        stable = stable_end(self._source)
        if stable > self._chunked:
            block = self._source[self._chunked : stable]
            if block.strip():
                self._chunks.append(self._render_source(block, width))
            self._chunked = stable
        parts = list(self._chunks)
        tail = self._source[self._chunked :]
        if tail.strip():
            parts.append(self._render_source(tail, width))
        return _BLOCK_GAP.join(parts)
    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        result = super().get_selection(selection)
        if result is None:
            return None
        text, ending = result
        # Code blocks pad every line to full width with background spaces.
        return "\n".join(line.rstrip() for line in text.split("\n")), ending
