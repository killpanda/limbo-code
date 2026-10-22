"""Direct tests for the GitignoreMatcher seam."""

from __future__ import annotations

from pathlib import Path

from limbo.tools.ignore import GitignoreMatcher


def make_matcher(tmp_path: Path, gitignore: str) -> GitignoreMatcher:
    (tmp_path / ".gitignore").write_text(gitignore)
    return GitignoreMatcher(tmp_path)


def test_negation_reincludes(tmp_path: Path):
    matcher = make_matcher(tmp_path, "*.log\n!keep.log\n")
    assert matcher.is_ignored("a.log")
    assert not matcher.is_ignored("keep.log")


