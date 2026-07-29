"""Direct tests for the GitignoreMatcher seam."""

from __future__ import annotations

from pathlib import Path

from limbo.tools.ignore import GitignoreMatcher


def make_matcher(tmp_path: Path, gitignore: str) -> GitignoreMatcher:
    (tmp_path / ".gitignore").write_text(gitignore)
    return GitignoreMatcher(tmp_path)


def test_no_gitignore_ignores_nothing(tmp_path: Path):
    matcher = GitignoreMatcher(tmp_path)
    assert not matcher.is_ignored("anything.py")


def test_simple_pattern_matches_at_any_depth(tmp_path: Path):
    matcher = make_matcher(tmp_path, "*.log\n")
    assert matcher.is_ignored("a.log")
    assert matcher.is_ignored("sub/dir/b.log")
    assert not matcher.is_ignored("a.py")


def test_directory_pattern_ignores_contents_but_not_file_of_same_name(tmp_path: Path):
    matcher = make_matcher(tmp_path, "build/\n")
    assert matcher.is_ignored("build/out.o")
    assert not matcher.is_ignored("build")


def test_anchored_pattern_only_matches_from_root(tmp_path: Path):
    matcher = make_matcher(tmp_path, "/dist\n")
    assert matcher.is_ignored("dist")
    assert not matcher.is_ignored("sub/dist")


def test_negation_reincludes(tmp_path: Path):
    matcher = make_matcher(tmp_path, "*.log\n!keep.log\n")
    assert matcher.is_ignored("a.log")
    assert not matcher.is_ignored("keep.log")


def test_comments_and_blank_lines_skipped(tmp_path: Path):
    matcher = make_matcher(tmp_path, "# comment\n\n*.pyc\n")
    assert matcher.is_ignored("x.pyc")
    assert len(matcher.rules) == 1


def test_doublestar_pattern(tmp_path: Path):
    matcher = make_matcher(tmp_path, "**/tmp/**\n")
    assert matcher.is_ignored("a/tmp/b.txt")
