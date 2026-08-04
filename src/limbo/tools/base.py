"""Base class and shared helpers for tools.

The base module owns the rituals every tool used to repeat: workdir-scoped
path resolution (raising ``ToolError`` instead of returning union types)
and the output truncation policy.

The workdir scope and the sensitive-file list are *convenience guardrails*,
not a security boundary: they keep the model from wandering outside the
project or pulling secrets into the context by accident, and the user can
widen the scope by mentioning paths in a message (session-scoped grants).
``bash`` is intentionally not covered by either guardrail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from limbo.models import ToolResult

MAX_OUTPUT_BYTES = 512 * 1024

# Dual-threshold truncation limits (whichever is hit first wins), aligned
# with pi's DEFAULT_MAX_LINES / DEFAULT_MAX_BYTES.
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50 KB


class ToolError(Exception):
    """Raised by tool helpers; ``BaseTool.execute`` turns it into an error result."""


class BaseTool(ABC):
    """Abstract base class for all Limbo tools.

    Subclasses implement ``run``; ``execute`` is the public entry point and
    converts ``ToolError`` into an error ``ToolResult`` uniformly.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def __init__(self, workdir: Path, allowed_roots: set[Path] | None = None):
        self.workdir = workdir.resolve()
        # Session-scoped extra roots the fence also accepts (e.g. paths a
        # real user mentioned in a submitted message). Shared by reference
        # with the ToolRegistry so late grants take effect immediately.
        self.allowed_roots: set[Path] = (
            allowed_roots if allowed_roots is not None else set()
        )

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            return self.run(arguments)
        except ToolError as e:
            return ToolResult(success=False, error=str(e))

    def cancel(self) -> None:
        """Best-effort cancellation of in-flight work (ESC interrupt).

        Default no-op: file tools always run to completion so history can
        record their real result — a "cancelled" worker thread would keep
        writing to disk while history claims the call never happened
        (RFC LIM-53). BashTool overrides this to kill its process tree.
        """

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        ...

    # -- path resolution -------------------------------------------------------

    def resolve(self, raw_path: str, *, strict: bool = True) -> Path:
        """Resolve ``raw_path`` against the workdir/allowed-roots guardrail.

        ``~`` is expanded and absolute paths are honored as-is; relative
        paths resolve under the workdir. Raises ``ToolError`` for paths
        outside the guardrail, unresolvable paths, and (when ``strict``)
        broken symlinks.
        """
        try:
            candidate = Path(raw_path).expanduser()
        except RuntimeError as e:  # home directory undeterminable
            raise ToolError(f"Invalid path: {e}") from e
        if not candidate.is_absolute():
            candidate = self.workdir / candidate
        try:
            target = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as e:
            raise ToolError(f"Invalid path: {e}") from e

        if not self.is_within_scope(target):
            raise ToolError(
                f"Path is outside working directory ({self.workdir}). "
                "This scope is a convenience guardrail, not a security "
                "boundary. If access is genuinely needed, ask the user to "
                "mention the path in a message to grant it."
            )

        if strict and candidate.is_symlink() and not target.exists():
            raise ToolError("Invalid path: broken symlink")

        return target

    def is_within_scope(self, path: Path) -> bool:
        """True if a resolved path is inside the workdir or an allowed root."""
        if is_within_workdir(path, self.workdir):
            return True
        # Snapshot: tools run in worker threads while the UI thread may add
        # grants; iterating the live set could raise "changed size during
        # iteration".
        return any(
            is_within_workdir(path, root) for root in tuple(self.allowed_roots)
        )

    def resolve_existing(
        self, raw_path: str, *, noun: str = "Path", kind: str = "any"
    ) -> Path:
        """Resolve a path that must exist. ``kind``: any | file | dir."""
        target = self.resolve(raw_path)
        if not target.exists():
            raise ToolError(f"{noun} not found: {raw_path}")
        if kind == "file" and not target.is_file():
            raise ToolError(f"Not a file: {raw_path}")
        if kind == "dir" and not target.is_dir():
            raise ToolError(f"Not a directory: {raw_path}")
        return target

    def resolve_creatable(self, raw_path: str) -> Path:
        """Resolve a path that may not exist yet (e.g. for writing)."""
        return self.resolve(raw_path, strict=False)

def is_sensitive_path(path: Path, sensitive_files: Collection[str]) -> bool:
    """True if ``path``'s name or any of its parts is on the sensitive list.

    Convenience guardrail against accidentally pulling secrets (``.env``,
    SSH keys) into the model context — not a security boundary (``bash``
    is not covered by it). Shared by ``read``/``grep``/``find`` so the
    guardrail behaves identically across file tools.
    """
    return any(part in sensitive_files for part in path.parts)


def is_within_workdir(path: Path, workdir: Path) -> bool:
    """Return True if resolved path is inside or equal to workdir."""
    try:
        path.resolve().relative_to(workdir.resolve())
        return True
    except ValueError:
        return False


def truncate_output(
    output: str, *, limit: int = MAX_OUTPUT_BYTES, suffix: str = "\n[Output truncated.]"
) -> str:
    """Truncate ``output`` to ``limit`` bytes (UTF-8 safe) and append ``suffix``."""
    encoded = output.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return output
    return encoded[:limit].decode("utf-8", errors="replace") + suffix


@dataclass(frozen=True)
class TruncationResult:
    """Structured result of ``truncate_head`` / ``truncate_tail``."""

    content: str
    truncated: bool
    truncated_by: str | None  # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    start_line: int  # 1-indexed first kept line (>1 only for tail truncation)
    edge_line_partial: bool  # the single kept edge line was byte-truncated


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def _split_lines(content: str) -> list[str]:
    """Split into lines without keeping separators; trailing newline ignored."""
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _head_bytes(text: str, max_bytes: int) -> str:
    """UTF-8-safe byte-prefix of ``text`` fitting ``max_bytes``."""
    return text.encode("utf-8", errors="replace")[:max_bytes].decode(
        "utf-8", errors="replace"
    )


def _tail_bytes(text: str, max_bytes: int) -> str:
    """UTF-8-safe byte-suffix of ``text`` fitting ``max_bytes``."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8", errors="replace")


def truncate_head(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the first ``max_lines``/``max_bytes`` of ``content`` (file reads).

    Never returns partial lines, except when the very first line alone
    exceeds ``max_bytes`` — then a byte-prefix of that line is kept and
    ``edge_line_partial`` is set.
    """
    total_bytes = _byte_len(content)
    lines = _split_lines(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            start_line=1,
            edge_line_partial=False,
        )

    kept: list[str] = []
    kept_bytes = 0
    truncated_by = "lines"
    partial = False
    for line in lines:
        if len(kept) >= max_lines:
            break
        # +1 for the newline joining this line to the previous kept one.
        line_bytes = _byte_len(line) + (1 if kept else 0)
        if kept_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not kept:
                kept.append(_head_bytes(line, max_bytes))
                partial = True
            break
        kept.append(line)
        kept_bytes += line_bytes

    return TruncationResult(
        content="\n".join(kept),
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept),
        start_line=1,
        edge_line_partial=partial,
    )


def truncate_tail(
    content: str,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the last ``max_lines``/``max_bytes`` of ``content`` (bash output).

    Errors and final results live at the end of command output, so the tail
    is what the model needs. Never returns partial lines, except when the
    very last line alone exceeds ``max_bytes`` — then a byte-suffix of that
    line is kept and ``edge_line_partial`` is set.
    """
    total_bytes = _byte_len(content)
    lines = _split_lines(content)
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            start_line=1,
            edge_line_partial=False,
        )

    kept: list[str] = []
    kept_bytes = 0
    truncated_by = "lines"
    partial = False
    for line in reversed(lines):
        if len(kept) >= max_lines:
            break
        # +1 for the newline joining this line to the next kept one.
        line_bytes = _byte_len(line) + (1 if kept else 0)
        if kept_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not kept:
                kept.insert(0, _tail_bytes(line, max_bytes))
                partial = True
            break
        kept.insert(0, line)
        kept_bytes += line_bytes

    return TruncationResult(
        content="\n".join(kept),
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept),
        start_line=total_lines - len(kept) + 1,
        edge_line_partial=partial,
    )
