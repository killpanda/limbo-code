"""Extract user-mentioned paths from a submitted message (implicit grants).

When a real user submits a message containing paths, the harness treats
those paths as user-attested authority and grants the file tools access to
them for the rest of the session. Only paths that actually exist are
grantable: a directory grants its subtree, a file grants just that file,
and non-existent paths are ignored (hallucinated paths and pasted junk
must not widen the fence).

The grant decision always happens here, in harness code, on the
human-submit event — never inside the agent loop from model text.
"""

from __future__ import annotations

import re
from pathlib import Path

# Quoted segments whose content starts like a path ("/abs", "~/...").
_QUOTED = re.compile(r"""["'](?P<path>(?:/|~/)[^"']+)["']""")

# Bare tokens starting with "/" or "~/". A path char class that stops at
# whitespace and the usual quoting/grouping characters; the lookbehind
# keeps us from matching inside longer tokens (URLs, relative paths).
_BARE = re.compile(r"(?<![\w~./:-])(?P<path>~/[^\s\"'`()\[\]{}<>]*|/[^\s\"'`()\[\]{}<>]*)")

# Punctuation that commonly hugs a path in prose but is not part of it.
_TRAILING = "\"'`).,;:!?]}>"


def extract_grantable_paths(text: str) -> list[Path]:
    """Return existing absolute paths mentioned in ``text``, deduplicated.

    A returned directory stands for its whole subtree, a file for itself.
    The filesystem root and other meaningless grants are skipped, as is any
    path already covered by an earlier (shorter) grant.
    """
    candidates: set[str] = set()
    for match in _QUOTED.finditer(text):
        candidates.add(match.group("path"))
    for match in _BARE.finditer(text):
        candidates.add(match.group("path"))

    grants: list[Path] = []
    # Shorter (parent) paths first so subsumed children drop out.
    for raw in sorted(candidates, key=len):
        cleaned = raw.rstrip(_TRAILING)
        if cleaned in ("", "/", "~"):
            continue
        try:
            resolved = Path(cleaned).expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if resolved == resolved.parent:  # filesystem root: would void the fence
            continue
        if not resolved.exists():
            continue
        if any(_is_within(resolved, grant) for grant in grants):
            continue
        grants.append(resolved)
    return grants


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
