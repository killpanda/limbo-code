"""Tell '/'-leading file paths apart from slash commands.

Typing an absolute path like ``/Users/admin/docs.pdf`` used to hit the
slash-command dispatcher and die with "未知命令". These heuristics let the
screen fall back to sending such input as a normal message and turn
existing files mentioned in it into attachments (auto-attach).
"""

from __future__ import annotations

from pathlib import Path

from limbo.models import Attachment
from limbo.tools.read import detect_image_mime


def looks_like_path(text: str) -> bool:
    """Whether '/'-leading text is more likely a file path than a command.

    True when the first token contains another '/' after the leading one
    (``/Users/admin/docs.pdf`` — command names never do), or when the
    token resolves to an existing file or directory (``/tmp``). Commands
    are checked before this heuristic, so a colliding name stays a command.
    """
    stripped = text.strip()
    token = stripped.split(maxsplit=1)[0] if stripped else text
    if "/" in token[1:]:
        return True
    return Path(token).expanduser().exists()


def attachments_from_text(text: str) -> list[Attachment]:
    """Auto-attach existing files referenced by absolute-path tokens.

    Only absolute (or ``~``-rooted) paths are considered: relative-looking
    tokens are too ambiguous with ordinary words. Directories and missing
    files stay plain text — the path itself remains in the message either
    way, so nothing is lost. Images are sniffed by magic bytes so they
    take the vision path (RFC v2 §4.3.1).
    """
    attachments: list[Attachment] = []
    seen: set[str] = set()
    for token in text.split():
        if not token.startswith(("/", "~")):
            continue
        path = Path(token).expanduser()
        if not path.is_absolute() or not path.is_file():
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        mime = detect_image_mime(path)
        if mime is not None:
            attachments.append(
                Attachment(kind="image", name=path.name, path=key, mime=mime)
            )
        else:
            attachments.append(Attachment(kind="file", name=path.name, path=key))
    return attachments
