"""Tests for implicit user-path grants (extract_grantable_paths)."""

from __future__ import annotations

from pathlib import Path

import pytest

from limbo.user_paths import extract_grantable_paths


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "proj" / "src").mkdir(parents=True)
    (tmp_path / "proj" / "src" / "main.py").write_text("x")
    (tmp_path / "notes.txt").write_text("y")
    return tmp_path


def test_extracts_absolute_dir(tree):
    grants = extract_grantable_paths(f"帮我看看 {tree}/proj 这个项目")
    assert grants == [(tree / "proj").resolve()]


def test_extracts_tilde_path(tree, monkeypatch):
    monkeypatch.setenv("HOME", str(tree))
    grants = extract_grantable_paths("看下 ~/proj/src 的代码")
    assert grants == [(tree / "proj" / "src").resolve()]


def test_extracts_quoted_path_with_spaces(tree):
    spaced = tree / "my dir"
    spaced.mkdir()
    grants = extract_grantable_paths(f'看看 "{spaced}" 里有什么')
    assert grants == [spaced.resolve()]


def test_ignores_nonexistent_paths(tree):
    assert extract_grantable_paths(f"打开 {tree}/nope/missing.txt") == []


def test_ignores_relative_paths_and_urls(tree):
    text = f"src/main.py 和 https://example.com/{tree.name}/proj 都不用管"
    assert extract_grantable_paths(text) == []


def test_strips_trailing_punctuation(tree):
    grants = extract_grantable_paths(f"读一下 {tree}/notes.txt.")
    assert grants == [(tree / "notes.txt").resolve()]


def test_skips_filesystem_root():
    assert extract_grantable_paths("从 / 开始找") == []


def test_tilde_slash_does_not_grant_home(tree, monkeypatch):
    """'~/' must not widen the fence to the whole home directory."""
    monkeypatch.setenv("HOME", str(tree))
    assert extract_grantable_paths("放在 ~/ 下就行") == []
    # Sanity: a real subdirectory under home still grants normally.
    assert extract_grantable_paths("看下 ~/proj") == [(tree / "proj").resolve()]


def test_strips_trailing_cjk_punctuation(tree):
    grants = extract_grantable_paths(f"路径是 {tree}/proj，请分析")
    assert grants == [(tree / "proj").resolve()]
    grants = extract_grantable_paths(f"读一下 {tree}/notes.txt。")
    assert grants == [(tree / "notes.txt").resolve()]
    grants = extract_grantable_paths(f"对比 {tree}/proj、{tree}/notes.txt；谢谢")
    assert set(grants) == {(tree / "proj").resolve(), (tree / "notes.txt").resolve()}


def test_child_subsumed_by_parent_grant(tree):
    grants = extract_grantable_paths(f"{tree}/proj 然后 {tree}/proj/src")
    assert grants == [(tree / "proj").resolve()]


def test_multiple_distinct_grants(tree):
    other = tree / "other"
    other.mkdir()
    grants = extract_grantable_paths(f"对比 {tree}/proj 和 {other}")
    assert set(grants) == {(tree / "proj").resolve(), other.resolve()}


def test_file_grants_itself(tree):
    grants = extract_grantable_paths(f"解释 {tree}/proj/src/main.py 这段")
    assert grants == [(tree / "proj" / "src" / "main.py").resolve()]
