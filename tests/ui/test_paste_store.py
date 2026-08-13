"""Pure unit tests for PasteStore (no Textual, no pilot).

The widget-level integration coverage lives in test_input_paste.py /
test_input_attachments.py; these hit the state machine's interface
directly.
"""

from __future__ import annotations

from pathlib import Path

from limbo.ui.paste_store import (
    PASTE_COLLAPSE_CHARS,
    PASTE_COLLAPSE_LINES,
    PasteStore,
    clean_pasted_text,
)


def big_lines(n: int = PASTE_COLLAPSE_LINES + 1) -> str:
    return "\n".join(f"line {i}" for i in range(n))


def big_chars(n: int = PASTE_COLLAPSE_CHARS + 1) -> str:
    return "x" * n


# -- clean_pasted_text -----------------------------------------------------------


def test_clean_normalizes_line_endings_and_strips_control_chars():
    assert clean_pasted_text("a\r\nb\rc") == "a\nb\nc"
    assert clean_pasted_text("a\x00\x07b") == "ab"
    assert clean_pasted_text("keep\ttabs\nand newlines") == "keep\ttabs\nand newlines"


# -- collapse ---------------------------------------------------------------------


def test_small_paste_passes_through_uncollapsed():
    store = PasteStore()
    assert store.collapse("hello") == "hello"
    assert store._pastes == {}


def test_collapse_by_line_count_uses_line_marker():
    store = PasteStore()
    marker = store.collapse(big_lines())
    assert marker == f"[粘贴的文本 #1，共 {PASTE_COLLAPSE_LINES + 1} 行]"


def test_collapse_by_char_count_uses_char_marker():
    store = PasteStore()
    marker = store.collapse(big_chars())
    assert marker == f"[粘贴的文本 #1，{PASTE_COLLAPSE_CHARS + 1} 字符]"


def test_collapse_ids_increment():
    store = PasteStore()
    m1 = store.collapse(big_chars())
    m2 = store.collapse(big_lines())
    assert "#1" in m1 and "#2" in m2


# -- expand -----------------------------------------------------------------------


def test_expand_roundtrips_stored_content():
    store = PasteStore()
    original = big_lines()
    marker = store.collapse(original)
    expanded, invalid = store.expand(f"前缀 {marker} 后缀")
    assert expanded == f"前缀 {original} 后缀"
    assert invalid == []


def test_expand_rejects_hand_typed_lookalike():
    """A marker never stored (hand-typed) stays literal and is reported."""
    store = PasteStore()
    text, invalid = store.expand("看看 [粘贴的文本 #9，共 99 行] 这个")
    assert text == "看看 [粘贴的文本 #9，共 99 行] 这个"
    assert invalid == [9]


def test_expand_rejects_marker_resurrected_after_clear():
    """Undo-after-clear revives marker text without content: invalid."""
    store = PasteStore()
    marker = store.collapse(big_chars())
    store.clear()
    text, invalid = store.expand(marker)
    assert text == marker
    assert invalid == [1]


def test_expand_handles_multiple_markers_independently():
    store = PasteStore()
    a, b = big_chars(), big_lines()
    ma, mb = store.collapse(a), store.collapse(b)
    expanded, invalid = store.expand(f"{ma}\n{mb}")
    assert expanded == f"{a}\n{b}"
    assert invalid == []


# -- atomic delete ------------------------------------------------------------------


def test_consume_marker_ending_at_returns_span_and_drops_content():
    store = PasteStore()
    marker = store.collapse(big_chars())
    line = f"abc {marker}"
    span = store.consume_marker_ending_at(line, len(line))
    assert span == (4, len(line))
    # Content is gone: a later expand treats the marker as invalid.
    assert store.expand(marker)[1] == [1]


def test_consume_marker_ending_at_misses_mid_marker_and_unknown_id():
    store = PasteStore()
    marker = store.collapse(big_chars())
    assert store.consume_marker_ending_at(marker, len(marker) - 1) is None
    unknown = "[粘贴的文本 #9，1 字符]"
    assert store.consume_marker_ending_at(unknown, len(unknown)) is None


def test_consume_marker_starting_at_roundtrip():
    store = PasteStore()
    marker = store.collapse(big_chars())
    span = store.consume_marker_starting_at(f"{marker} tail", 0)
    assert span == (0, len(marker))
    assert store._pastes == {}


# -- attachments -------------------------------------------------------------------


def test_clipboard_image_marker_and_live_filter(tmp_path):
    store = PasteStore()
    attachment, marker = store.add_clipboard_image(tmp_path / "x.png", "png")
    assert attachment.kind == "image"
    assert attachment.mime == "image/png"
    assert attachment.name == "剪贴板图片-1.png"
    assert marker == "[图片 #1]"
    # Marker present in the text → attachment rides along; deleted → not.
    assert store.live_attachments(f"看这里 {marker}") == [attachment]
    assert store.live_attachments("看这里") == []


def test_file_attachment_marker_format():
    store = PasteStore()
    attachment, marker = store.add_file(Path("/tmp/report.pdf"))
    assert attachment.kind == "file"
    assert attachment.name == "report.pdf"
    assert marker == "[文件 #1: report.pdf]"


def test_attachment_numbers_increment_across_kinds():
    store = PasteStore()
    _, m1 = store.add_file(Path("/a.txt"))
    _, m2 = store.add_clipboard_image("/p.png", "png")
    assert "#1" in m1 and "#2" in m2


# -- lifecycle ----------------------------------------------------------------------


def test_clear_resets_pastes_and_attachments():
    store = PasteStore()
    store.collapse(big_chars())
    store.add_file(Path("/a.txt"))
    store.clear()
    assert store._pastes == {}
    assert store.live_attachments("[文件 #1: a.txt]") == []
