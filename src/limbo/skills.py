"""Skill support: discover and load SKILL.md skills.

A skill is a directory containing a ``SKILL.md`` file with YAML-ish
frontmatter (``name``, ``description``) followed by the instruction body —
the same convention used by Claude Code and pi.

Discovery paths (later sources win on name collisions):
- user:    ``~/.limbo/skills/<name>/SKILL.md``
- project: ``<workdir>/.agents/skills/<name>/SKILL.md``

Skills are invoked through slash commands: ``/<name> [args]`` injects the
skill body (plus args) as the turn's prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_USER_SKILLS_DIR = Path.home() / ".limbo" / "skills"
PROJECT_SKILLS_DIR = ".agents/skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


def parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str]:
    """Parse SKILL.md content into (name, description, body).

    Frontmatter is a simple ``---`` delimited block of ``key: value`` lines.
    Without frontmatter, the fallback name is used and the whole file is the
    body.
    """
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            header = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            name = fallback_name
            description = ""
            for line in header.splitlines():
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                key = key.strip()
                value = value.strip()
                if key == "name" and value:
                    name = value
                elif key == "description":
                    description = value
            return name, description, body
    return fallback_name, "", text


def _load_skills_from(skills_dir: Path) -> list[Skill]:
    skills: list[Skill] = []
    if not skills_dir.is_dir():
        return skills
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name, description, body = parse_skill_md(text, fallback_name=path.parent.name)
        skills.append(
            Skill(name=name, description=description, body=body, path=path)
        )
    return skills


def discover_skills(
    workdir: Path, user_dir: Path | None = None
) -> list[Skill]:
    """Discover skills from the user dir and the project (.agents/skills).

    Project skills override user skills with the same name. Sorted by name.
    """
    user_dir = user_dir if user_dir is not None else DEFAULT_USER_SKILLS_DIR
    by_name: dict[str, Skill] = {}
    for skill in _load_skills_from(user_dir):
        by_name[skill.name] = skill
    for skill in _load_skills_from(workdir / PROJECT_SKILLS_DIR):
        by_name[skill.name] = skill
    return [by_name[name] for name in sorted(by_name)]
