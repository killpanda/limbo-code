"""Skill support: discover and load SKILL.md skills.

A skill is a directory containing a ``SKILL.md`` file with YAML-ish
frontmatter (``name``, ``description``) followed by the instruction body —
the same convention used by Claude Code and pi.

Discovery paths (later sources win on name collisions):
- user:    ``~/.limbo/skills/<name>/SKILL.md``
- global:  ``~/.agents/skills/<name>/SKILL.md``
- project: ``<workdir>/.agents/skills/<name>/SKILL.md``

Skills are exposed to the model two ways (dual track):

- ``<available_skills>`` catalog in the system prompt (name, description
  and location only — progressive disclosure). The model loads the full
  SKILL.md with the read tool when a task matches a description. Skills
  with ``disable-model-invocation: true`` are excluded from the catalog.
- Slash commands: ``/<name> [args]`` injects the skill body (plus args)
  as the turn's prompt — the explicit fallback.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_USER_SKILLS_DIR = Path.home() / ".limbo" / "skills"
GLOBAL_AGENTS_SKILLS_DIR = Path.home() / ".agents" / "skills"
PROJECT_SKILLS_DIR = ".agents/skills"

# Validation limits per the Agent Skills spec (agentskills.io).
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
_SKILL_NAME_RE = re.compile(r"^[a-z0-9-]+$")


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    disable_model_invocation: bool = False


def parse_skill_md(text: str, fallback_name: str) -> tuple[str, str, str, bool]:
    """Parse SKILL.md content into (name, description, body, disable_model_invocation).

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
            disable_model_invocation = False
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
                elif key == "disable-model-invocation":
                    disable_model_invocation = value.lower() in {"true", "yes", "1"}
            return name, description, body, disable_model_invocation
    return fallback_name, "", text, False


def validate_skill(name: str, description: str) -> list[str]:
    """Validate skill name/description per the Agent Skills spec.

    Returns a list of error messages (empty when valid).
    """
    errors: list[str] = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters")
    if not _SKILL_NAME_RE.match(name):
        errors.append("name must be lowercase a-z, 0-9 and hyphens only")
    elif name.startswith("-") or name.endswith("-") or "--" in name:
        errors.append("name must not start/end with a hyphen or contain '--'")
    if not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters")
    return errors


def escape_xml(text: str) -> str:
    """Escape the five XML predefined entities."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format the ``<available_skills>`` catalog for the system prompt.

    Only name/description/location are included — progressive disclosure:
    the model loads the full SKILL.md with the read tool on demand. Skills
    with ``disable_model_invocation`` are excluded (slash-command only).
    Returns an empty string when no skill is model-invocable.
    """
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""
    lines = [
        "",
        "",
        "The following skills provide specialized instructions for specific tasks. "
        "When a task matches a skill's description, use the read tool to load the "
        "skill's file. When a skill file references a relative path, resolve it "
        "against the skill directory (parent of SKILL.md / dirname of the path) "
        "and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{escape_xml(str(skill.path))}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def _load_skills_from(skills_dir: Path) -> list[Skill]:
    skills: list[Skill] = []
    if not skills_dir.is_dir():
        return skills
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        name, description, body, disable_model_invocation = parse_skill_md(
            text, fallback_name=path.parent.name
        )
        errors = validate_skill(name, description)
        if errors:
            logger.warning("Skipping skill %s: %s", path, "; ".join(errors))
            continue
        skills.append(
            Skill(
                name=name,
                description=description,
                body=body,
                path=path,
                disable_model_invocation=disable_model_invocation,
            )
        )
    return skills


def discover_skills(
    workdir: Path, user_dir: Path | None = None, global_dir: Path | None = None
) -> list[Skill]:
    """Discover skills from the user dir, the global agents dir and the project.

    Sources in ascending precedence: ``~/.limbo/skills``, ``~/.agents/skills``,
    ``<workdir>/.agents/skills``. Later sources win name collisions (a warning
    is logged for each). Sorted by name.
    """
    user_dir = user_dir if user_dir is not None else DEFAULT_USER_SKILLS_DIR
    global_dir = (
        global_dir if global_dir is not None else GLOBAL_AGENTS_SKILLS_DIR
    )
    by_name: dict[str, Skill] = {}
    for skill in (
        _load_skills_from(user_dir)
        + _load_skills_from(global_dir)
        + _load_skills_from(workdir / PROJECT_SKILLS_DIR)
    ):
        if skill.name in by_name:
            logger.warning(
                "Skill name collision: %r from %s overrides %s",
                skill.name,
                skill.path,
                by_name[skill.name].path,
            )
        by_name[skill.name] = skill
    return [by_name[name] for name in sorted(by_name)]
