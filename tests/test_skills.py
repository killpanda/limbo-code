"""Tests for skill discovery (limbo.skills)."""

from __future__ import annotations

from pathlib import Path

from limbo.skills import discover_skills, parse_skill_md


def write_skill(
    base: Path, dirname: str, content: str, filename: str = "SKILL.md"
) -> Path:
    skill_dir = base / dirname
    skill_dir.mkdir(parents=True)
    path = skill_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_frontmatter():
    text = "---\nname: tdd\ndescription: Test-driven development.\n---\n\n# TDD\n\nDo it.\n"
    name, description, body = parse_skill_md(text, fallback_name="x")

    assert name == "tdd"
    assert description == "Test-driven development."
    assert body.strip().startswith("# TDD")


def test_parse_description_with_colon():
    text = "---\nname: x\ndescription: note: uses a colon\n---\nbody\n"
    _, description, _ = parse_skill_md(text, fallback_name="x")
    assert description == "note: uses a colon"


def test_parse_without_frontmatter_uses_fallback():
    name, description, body = parse_skill_md("# Just markdown\n", fallback_name="raw")

    assert name == "raw"
    assert description == ""
    assert body.strip() == "# Just markdown"


def test_discover_from_user_and_project_dirs(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(user_dir, "aaa", "---\nname: aaa\n---\nuser skill\n")
    write_skill(project_dir, "bbb", "---\nname: bbb\n---\nproject skill\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=user_dir
    )

    assert {s.name for s in skills} == {"aaa", "bbb"}


def test_project_skill_overrides_user_skill(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(user_dir, "dup", "---\nname: dup\n---\nuser version\n")
    write_skill(project_dir, "dup", "---\nname: dup\n---\nproject version\n")

    skills = discover_skills(tmp_path / "project", user_dir=user_dir)

    assert len(skills) == 1
    assert skills[0].body.strip() == "project version"


def test_discover_ignores_dirs_without_skill_md(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    (user_dir / "not-a-skill").mkdir(parents=True)
    (user_dir / "not-a-skill" / "readme.txt").write_text("nope")
    write_skill(user_dir, "real", "---\nname: real\n---\nok\n")

    skills = discover_skills(tmp_path / "project", user_dir=user_dir)

    assert [s.name for s in skills] == ["real"]


def test_discover_missing_dirs_returns_empty(tmp_path: Path):
    assert discover_skills(tmp_path / "nope", user_dir=tmp_path / "gone") == []


def test_skill_keeps_path_for_relative_resolution(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    path = write_skill(user_dir, "tdd", "---\nname: tdd\n---\nsee tests.md\n")

    skills = discover_skills(tmp_path / "project", user_dir=user_dir)

    assert skills[0].path == path
