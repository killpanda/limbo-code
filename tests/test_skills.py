"""Tests for skill discovery (limbo.skills)."""

from __future__ import annotations

import logging
from pathlib import Path

from limbo.skills import (
    Skill,
    discover_skills,
    escape_xml,
    format_skills_for_prompt,
    parse_skill_md,
)


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
    name, description, body, disable = parse_skill_md(text, fallback_name="x")

    assert name == "tdd"
    assert description == "Test-driven development."
    assert body.strip().startswith("# TDD")
    assert disable is False


def test_parse_description_with_colon():
    text = "---\nname: x\ndescription: note: uses a colon\n---\nbody\n"
    _, description, _, _ = parse_skill_md(text, fallback_name="x")
    assert description == "note: uses a colon"


def test_parse_disable_model_invocation():
    text = "---\nname: x\ndescription: d\ndisable-model-invocation: true\n---\nbody\n"
    *_, disable = parse_skill_md(text, fallback_name="x")
    assert disable is True


def test_parse_without_frontmatter_uses_fallback():
    name, description, body, disable = parse_skill_md("# Just markdown\n", fallback_name="raw")

    assert name == "raw"
    assert description == ""
    assert body.strip() == "# Just markdown"
    assert disable is False


def test_discover_from_user_and_project_dirs(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    global_dir = tmp_path / "global_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(user_dir, "aaa", "---\nname: aaa\ndescription: a\n---\nuser skill\n")
    write_skill(project_dir, "bbb", "---\nname: bbb\ndescription: b\n---\nproject skill\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=user_dir, global_dir=global_dir
    )

    assert {s.name for s in skills} == {"aaa", "bbb"}


def test_discover_from_global_agents_dir(tmp_path: Path):
    global_dir = tmp_path / "global_skills"
    write_skill(global_dir, "ccc", "---\nname: ccc\ndescription: c\n---\nglobal skill\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=tmp_path / "user_skills", global_dir=global_dir
    )

    assert {s.name for s in skills} == {"ccc"}


def test_global_skill_overrides_user_skill(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    global_dir = tmp_path / "global_skills"
    write_skill(user_dir, "dup", "---\nname: dup\ndescription: d\n---\nuser version\n")
    write_skill(global_dir, "dup", "---\nname: dup\ndescription: d\n---\nglobal version\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=user_dir, global_dir=global_dir
    )

    assert len(skills) == 1
    assert skills[0].body.strip() == "global version"


def test_project_skill_overrides_user_skill(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    global_dir = tmp_path / "global_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(user_dir, "dup", "---\nname: dup\ndescription: d\n---\nuser version\n")
    write_skill(project_dir, "dup", "---\nname: dup\ndescription: d\n---\nproject version\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=user_dir, global_dir=global_dir
    )

    assert len(skills) == 1
    assert skills[0].body.strip() == "project version"


def test_project_skill_overrides_global_skill(tmp_path: Path):
    global_dir = tmp_path / "global_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(global_dir, "dup", "---\nname: dup\ndescription: d\n---\nglobal version\n")
    write_skill(project_dir, "dup", "---\nname: dup\ndescription: d\n---\nproject version\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=tmp_path / "user_skills", global_dir=global_dir
    )

    assert len(skills) == 1
    assert skills[0].body.strip() == "project version"


def test_skill_name_collision_logs_warning(tmp_path: Path, caplog):
    user_dir = tmp_path / "user_skills"
    project_dir = tmp_path / "project" / ".agents" / "skills"
    write_skill(user_dir, "dup", "---\nname: dup\ndescription: d\n---\nuser version\n")
    write_skill(project_dir, "dup", "---\nname: dup\ndescription: d\n---\nproject version\n")

    with caplog.at_level(logging.WARNING, logger="limbo.skills"):
        discover_skills(
            tmp_path / "project", user_dir=user_dir, global_dir=tmp_path / "global_skills"
        )

    assert any("collision" in r.message and "dup" in r.message for r in caplog.records)


def test_discover_skips_invalid_skills_with_warning(tmp_path: Path, caplog):
    user_dir = tmp_path / "user_skills"
    write_skill(user_dir, "no-desc", "---\nname: no-desc\n---\nbody\n")
    write_skill(user_dir, "Bad_Name", "---\nname: Bad_Name\ndescription: d\n---\nbody\n")
    write_skill(user_dir, "-lead", "---\nname: -lead\ndescription: d\n---\nbody\n")
    write_skill(user_dir, "good", "---\nname: good\ndescription: d\n---\nbody\n")

    with caplog.at_level(logging.WARNING, logger="limbo.skills"):
        skills = discover_skills(
            tmp_path / "project", user_dir=user_dir, global_dir=tmp_path / "global_skills"
        )

    assert [s.name for s in skills] == ["good"]
    messages = "\n".join(r.message for r in caplog.records)
    assert "description is required" in messages
    assert "lowercase" in messages
    assert "hyphen" in messages


def test_discover_ignores_dirs_without_skill_md(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    (user_dir / "not-a-skill").mkdir(parents=True)
    (user_dir / "not-a-skill" / "readme.txt").write_text("nope")
    write_skill(user_dir, "real", "---\nname: real\ndescription: d\n---\nok\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=user_dir, global_dir=tmp_path / "global_skills"
    )

    assert [s.name for s in skills] == ["real"]


def test_discover_missing_dirs_returns_empty(tmp_path: Path):
    assert discover_skills(
        tmp_path / "nope",
        user_dir=tmp_path / "gone",
        global_dir=tmp_path / "also-gone",
    ) == []


def test_skill_keeps_path_for_relative_resolution(tmp_path: Path):
    user_dir = tmp_path / "user_skills"
    path = write_skill(user_dir, "tdd", "---\nname: tdd\ndescription: d\n---\nsee tests.md\n")

    skills = discover_skills(
        tmp_path / "project", user_dir=user_dir, global_dir=tmp_path / "global_skills"
    )

    assert skills[0].path == path


def _skill(name="tdd", description="TDD workflow.", disable=False, path="/x/tdd/SKILL.md"):
    return Skill(
        name=name,
        description=description,
        body="",
        path=Path(path),
        disable_model_invocation=disable,
    )


def test_format_skills_for_prompt_includes_catalog_only():
    skill = _skill()
    block = format_skills_for_prompt([skill])

    assert "<available_skills>" in block
    assert "<name>tdd</name>" in block
    assert "<description>TDD workflow.</description>" in block
    assert "<location>/x/tdd/SKILL.md</location>" in block
    assert "read tool" in block
    # Body must not be inlined: progressive disclosure.
    full_block = format_skills_for_prompt([
        Skill(name="tdd", description="TDD workflow.", body="FULL BODY TEXT", path=Path("/x"))
    ])
    assert "FULL BODY TEXT" not in full_block


def test_format_skills_for_prompt_excludes_disable_model_invocation():
    block = format_skills_for_prompt([
        _skill(name="visible"),
        _skill(name="hidden", disable=True, path="/x/hidden/SKILL.md"),
    ])

    assert "visible" in block
    assert "hidden" not in block


def test_format_skills_for_prompt_empty_when_all_disabled():
    assert format_skills_for_prompt([_skill(disable=True)]) == ""
    assert format_skills_for_prompt([]) == ""


def test_format_skills_for_prompt_escapes_xml():
    block = format_skills_for_prompt([
        _skill(description="use <tag> & 'quotes' \"here\"")
    ])

    assert "<tag>" not in block.replace("<skill>", "").replace("</skill>", "")
    assert "&lt;tag&gt; &amp; &apos;quotes&apos; &quot;here&quot;" in block


def test_escape_xml():
    assert escape_xml("a<b>&c'd\"e") == "a&lt;b&gt;&amp;c&apos;d&quot;e"
