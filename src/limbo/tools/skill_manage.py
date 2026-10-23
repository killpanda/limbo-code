"""The ``skill_manage`` tool (daily mode): the agent curates its own skills.

Skills are procedural memory ("how to do this class of task for this
user"); ``memory`` holds declarative facts. Every write lands in the daily
skills dir (``~/.limbo/daily/skills``): a bundled, user or project skill
being changed is first copied there (copy-on-write), and the daily copy
then wins by name — so the agent never edits the package, the user's
coding skills or a repository.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from limbo.models import ToolResult
from limbo.skills import Skill, discover_skills, parse_skill_md, validate_skill
from limbo.tools.base import BaseTool, ToolError
from limbo.tools.mutation_queue import mutation_lock_for

SUPPORT_DIRS = ("references", "templates", "scripts")
ACTIONS = ("create", "patch", "write_file")

_DESCRIPTION = (
    "Create or improve skills: your procedural memory for recurring kinds of "
    "task (how to do X for this user). One call takes an operations array and "
    "applies it all-or-nothing. Ops: create (a whole new SKILL.md with "
    "frontmatter `name` and `description`), patch (replace old_string with "
    "new_string in a skill's SKILL.md, or in file_path; read the file first), "
    "write_file (add or replace a support file under references/, templates/ "
    "or scripts/). Skills are class-level (e.g. `weekly-report`, never "
    "`fix-bug-123`). Write lessons, not logs: an imperative rule plus why, one "
    "rule per bullet, no dates or session narration. The description is the "
    "trigger: say when to use the skill in its first sentence. Skills cannot "
    "be deleted with this tool."
)


class SkillManageTool(BaseTool):
    name = "skill_manage"
    description = _DESCRIPTION
    parameters = {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "name": {"type": "string", "description": "The skill's name."},
                        "content": {
                            "type": "string",
                            "description": "create: the full SKILL.md. write_file: the file.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "patch: exact text to replace (must occur once).",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "patch: the replacement (empty deletes).",
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "write_file / patch: path relative to the skill dir, "
                                "e.g. references/style.md (patch defaults to SKILL.md)."
                            ),
                        },
                    },
                    "required": ["action", "name"],
                },
            },
        },
        "required": ["operations"],
    }

    def __init__(self, workdir: Path, skills_dir: Callable[[], Path]):
        super().__init__(workdir)
        # Resolved per call (tests redirect the module default).
        self._skills_dir = skills_dir

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        operations = arguments.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ToolError("operations must be a non-empty list")
        root = self._skills_dir()
        skills = {s.name: s for s in discover_skills(self.workdir, daily=True)}
        # Plan everything in memory first: nothing is written unless every
        # operation is valid (all-or-nothing).
        staged: dict[Path, str] = {}
        copies: dict[str, Path] = {}  # skill name -> source dir to copy first
        created: list[str] = []
        for i, op in enumerate(operations, 1):
            where = f"operation {i}: " if len(operations) > 1 else ""
            try:
                self._plan(op, root, skills, staged, copies, created)
            except ToolError as e:
                raise ToolError(f"{where}{e}") from e
        for name, source in copies.items():
            target = root / name
            if not target.exists():
                shutil.copytree(source, target)
        for path, text in staged.items():
            _write(path, text)
        touched = sorted({p.relative_to(root).parts[0] for p in staged})
        lines = [f"Updated skills: {', '.join(touched)} (in {root})."]
        copied = [n for n in copies if n not in created]
        if copied:
            lines.append(
                f"Copied {', '.join(copied)} into the daily skills dir first; "
                "the copy now replaces the original by name."
            )
        lines.append("Changes apply the next time a skill is loaded.")
        return ToolResult(success=True, output="\n".join(lines))

    def _plan(
        self,
        op: Any,
        root: Path,
        skills: dict[str, Skill],
        staged: dict[Path, str],
        copies: dict[str, Path],
        created: list[str],
    ) -> None:
        if not isinstance(op, dict):
            raise ToolError("each operation must be an object")
        action, name = op.get("action"), op.get("name")
        if action not in ACTIONS:
            raise ToolError(f"action must be one of: {', '.join(ACTIONS)}")
        if not isinstance(name, str) or validate_skill(name, "x"):
            raise ToolError(f"invalid skill name {name!r} (lowercase letters, digits, hyphens)")
        skill_dir = root / name
        if action == "create":
            if name in skills or name in created:
                raise ToolError(f"skill {name!r} already exists; patch it instead")
            content = _require_str(op, "content")
            _check_skill_md(content, name)
            staged[skill_dir / "SKILL.md"] = content
            created.append(name)
            return
        existing = skills.get(name)
        if existing is None and name not in created:
            raise ToolError(f"no skill named {name!r}; create it first")
        if existing is not None and existing.path.parent != skill_dir:
            copies.setdefault(name, existing.path.parent)
        rel = _relative(op.get("file_path") or "SKILL.md", allow_skill_md=action == "patch")
        target = skill_dir / rel
        if action == "write_file":
            staged[target] = _require_str(op, "content")
            return
        # patch: against the staged text, the daily copy, or the source.
        if target in staged:
            current = staged[target]
        else:
            source = target if target.exists() else (
                (existing.path.parent / rel) if existing is not None else target
            )
            try:
                current = source.read_text(encoding="utf-8")
            except OSError as e:
                raise ToolError(f"cannot read {rel} of {name!r}: {e}") from e
        old = _require_str(op, "old_string")
        new = op.get("new_string", "")
        if not isinstance(new, str):
            raise ToolError("new_string must be a string")
        count = current.count(old)
        if count != 1:
            raise ToolError(
                f"old_string occurs {count} times in {rel} of {name!r}; it must occur once"
            )
        updated = current.replace(old, new)
        if rel == PurePosixPath("SKILL.md"):
            _check_skill_md(updated, name)
        staged[target] = updated


def _require_str(op: dict[str, Any], key: str) -> str:
    value = op.get(key)
    if not isinstance(value, str) or not value:
        raise ToolError(f"{op.get('action')} needs {key}")
    return value


def _relative(raw: Any, *, allow_skill_md: bool) -> PurePosixPath:
    if not isinstance(raw, str):
        raise ToolError("file_path must be a string")
    rel = PurePosixPath(raw)
    if rel == PurePosixPath("SKILL.md") and allow_skill_md:
        return rel
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) < 2:
        raise ToolError(
            f"file_path must be relative and start with one of: {', '.join(SUPPORT_DIRS)}/"
        )
    if rel.parts[0] not in SUPPORT_DIRS:
        raise ToolError(f"file_path must start with one of: {', '.join(SUPPORT_DIRS)}/")
    return rel


def _check_skill_md(text: str, name: str) -> None:
    parsed_name, description, _, _ = parse_skill_md(text, fallback_name="")
    if not text.startswith("---") or not parsed_name:
        raise ToolError("SKILL.md must start with --- frontmatter holding name and description")
    if parsed_name != name:
        raise ToolError(f"frontmatter name {parsed_name!r} does not match {name!r}")
    errors = validate_skill(parsed_name, description)
    if errors:
        raise ToolError("; ".join(errors))


def _write(path: Path, text: str) -> None:
    with mutation_lock_for(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
