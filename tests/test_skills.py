"""Tests for skill discovery (limbo.skills)."""

from __future__ import annotations

from pathlib import Path

from limbo.skills import (
    discover_skills,
)


def write_skill(
    base: Path, dirname: str, content: str, filename: str = "SKILL.md"
) -> Path:
    skill_dir = base / dirname
    skill_dir.mkdir(parents=True)
    path = skill_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_project_skill_overrides_user_skill(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(user_dir, "dup", "---\nname: dup\ndescription: d\n---\nuser version\n")
    write_skill(project_dir, "dup", "---\nname: dup\ndescription: d\n---\nproject version\n")

    skills = discover_skills(tmp_path / "project", user_dir=user_dir)

    assert len(skills) == 1
    assert skills[0].body.strip() == "project version"


