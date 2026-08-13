"""Collapsed-paste and attachment-marker store (pure text state machine).

Big pastes collapse into one-line placeholder markers whose content lives
here; at submit time markers expand back, and markers without stored
content (hand-typed lookalikes, undo-resurrected after deletion) fail
validation and stay literal so a paste never vanishes silently.

No Textual imports: the input widget is an adapter translating key,
cursor, and clipboard events into store calls. The marker TEXT FORMAT is
owned here — both generation and the regex — so the two can never drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from limbo.models import Attachment

# Large pastes (above either threshold, pi-style) are collapsed into a
# one-line placeholder; the real content lives in the store and is
# expanded back at submit time.
PASTE_COLLAPSE_LINES = 10
PASTE_COLLAPSE_CHARS = 1000

# Matches paste placeholders like ``[粘贴的文本 #1，共 123 行]`` or
# ``[粘贴的文本 #1，1234 字符]``. Anchored to the exact formats the store
# inserts (no newlines, no free-form middle) so similar-looking pasted
# content is not misdetected.
PASTE_MARKER_RE = re.compile(r"\[粘贴的文本 #(\d+)(?:，共 \d+ 行|，\d+ 字符)\]")


def clean_pasted_text(text: str) -> str:
    """Normalize line endings and strip control chars (keep \\n and \\t)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(c for c in text if c in "\n\t" or ord(c) >= 32)


class PasteStore:
    """Owns collapsed-paste content and attachment markers for one input.

    Lifecycle is per-message: the widget calls :meth:`clear` on the same
    path that clears the document, so stale ids never leak into the next
    message.
    """

    def __init__(self) -> None:
        self._pastes: dict[int, str] = {}
        self._paste_counter = 0
        self._attachments: list[tuple[str, Attachment]] = []
        self._attachment_counter = 0

    # -- collapsed pastes -------------------------------------------------------

    def collapse(self, text: str) -> str:
        """Store an oversized paste and return its one-line marker.

        Small pastes are returned unchanged (inserted inline by the
        caller). The stored content is keyed by the marker's id; only ids
        still stored here expand at submit time (cf. pi's validPasteIds),
        so lookalike text is never expanded by accident.
        """
        lines = text.split("\n")
        if len(lines) <= PASTE_COLLAPSE_LINES and len(text) <= PASTE_COLLAPSE_CHARS:
            return text
        self._paste_counter += 1
        paste_id = self._paste_counter
        self._pastes[paste_id] = text
        if len(lines) > PASTE_COLLAPSE_LINES:
            return f"[粘贴的文本 #{paste_id}，共 {len(lines)} 行]"
        return f"[粘贴的文本 #{paste_id}，{len(text)} 字符]"

    def expand(self, text: str) -> tuple[str, list[int]]:
        """Replace markers with their stored content.

        Returns the expanded text plus the ids of markers whose content is
        no longer stored (deleted then restored via undo, or hand-typed
        lookalikes); those stay as literal text and are reported so the
        loss is surfaced instead of the paste vanishing silently.
        """
        invalid: list[int] = []

        def repl(match: re.Match[str]) -> str:
            paste_id = int(match.group(1))
            if paste_id in self._pastes:
                return self._pastes[paste_id]
            invalid.append(paste_id)
            return match.group(0)

        return PASTE_MARKER_RE.sub(repl, text), invalid

    def consume_marker_ending_at(self, line: str, col: int) -> tuple[int, int] | None:
        """A stored marker ending exactly at ``col``: drop it, return its span.

        The widget deletes the returned span atomically (backspace over a
        placeholder); dropping the stored content here keeps text and
        storage in lockstep — a later undo resurrects only inert text.
        """
        for match in PASTE_MARKER_RE.finditer(line):
            if match.end() == col and int(match.group(1)) in self._pastes:
                del self._pastes[int(match.group(1))]
                return match.start(), match.end()
        return None

    def consume_marker_starting_at(self, line: str, col: int) -> tuple[int, int] | None:
        """Delete-key counterpart of :meth:`consume_marker_ending_at`."""
        match = PASTE_MARKER_RE.match(line, col)
        if match is not None and int(match.group(1)) in self._pastes:
            del self._pastes[int(match.group(1))]
            return match.start(), match.end()
        return None

    # -- attachments --------------------------------------------------------------

    def add_clipboard_image(self, path: str | Path, ext: str) -> tuple[Attachment, str]:
        """Register a clipboard image; return (attachment, document marker)."""
        self._attachment_counter += 1
        n = self._attachment_counter
        attachment = Attachment(
            kind="image",
            name=f"剪贴板图片-{n}.{ext}",
            path=str(path),
            mime=f"image/{ext}",
        )
        marker = f"[图片 #{n}]"
        self._attachments.append((marker, attachment))
        return attachment, marker

    def add_file(self, path: Path) -> tuple[Attachment, str]:
        """Register a file attachment; return (attachment, document marker)."""
        self._attachment_counter += 1
        n = self._attachment_counter
        attachment = Attachment(kind="file", name=path.name, path=str(path))
        marker = f"[文件 #{n}: {path.name}]"
        self._attachments.append((marker, attachment))
        return attachment, marker

    def live_attachments(self, text: str) -> list[Attachment]:
        """Attachments whose marker survived editing in the submitted text."""
        return [a for marker, a in self._attachments if marker in text]

    # -- lifecycle ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset per-message state (called on the widget's clear() path)."""
        self._pastes.clear()
        self._attachments.clear()
