"""System-prompt assembly.

The ToolRegistry is the single source of truth for the tools section: a
tool that is not registered (e.g. ``bash_enabled = false``) never appears
in the prompt, and tool descriptions live in exactly one place. Project
and global ``AGENTS.md`` files and the skills catalog are appended as
context blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from limbo.skills import discover_skills, format_skills_for_prompt

if TYPE_CHECKING:
    from limbo.tools.registry import ToolRegistry

_BASE_INTRO = (
    "You are an expert coding assistant operating inside limbo, "
    "a coding agent harness. You help users by reading files, "
    "executing commands, editing code, and writing new files.\n\n"
)

_GUIDELINES_TAIL = (
    "- Use read to examine files before editing\n"
    "- Use edit for precise changes (edits[].old_text must match exactly)\n"
    "- Use write only for new files or complete rewrites\n"
    "- Be concise in your responses\n"
    "- Show file paths clearly when working with files"
)


def _first_sentence(description: str) -> str:
    """The summary line for a tool: its description's first sentence."""
    return description.split(". ")[0].rstrip(".")


def _tools_section(registry: ToolRegistry) -> str:
    """Available-tools list derived from the registry (never hand-maintained)."""
    lines = ["Available tools:"]
    for definition in registry.definitions():
        function = definition["function"]
        lines.append(f"- {function['name']}: {_first_sentence(function['description'])}")
    return "\n".join(lines)


def _guidelines(registry: ToolRegistry) -> str:
    lines = ["Guidelines:"]
    if registry.get("bash") is not None:
        lines.append("- Prefer grep/find/ls tools over bash for file exploration")
    lines.append(_GUIDELINES_TAIL)
    return "\n".join(lines)


def _context_block(workdir: Path, home: Path) -> str | None:
    """Project + global AGENTS.md files, wrapped in XML boundary tags."""
    context_files: list[tuple[str, str]] = []
    for candidate in (workdir / "AGENTS.md", home / ".limbo" / "AGENTS.md"):
        if candidate.is_file():
            try:
                context_files.append((str(candidate), candidate.read_text(encoding="utf-8")))
            except OSError:
                pass
    if not context_files:
        return None
    block = "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
    for file_path, content in context_files:
        block += (
            f'<project_instructions path="{file_path}">\n\n'
            f"{content}\n\n</project_instructions>\n\n"
        )
    block += "</project_context>\n"
    return block


def build_system_prompt(
    registry: ToolRegistry,
    workdir: Path,
    home: Path | None = None,
) -> str:
    """Assemble the system prompt: base text + tools + guidelines + context.

    The tools section and the bash guideline are derived from ``registry``,
    so the prompt always matches the tools the model can actually call.
    The skills catalog is included only when the read tool is available
    (the model loads SKILL.md files with read on demand).
    """
    parts = [
        _BASE_INTRO,
        _tools_section(registry),
        "\n\n",
        _guidelines(registry),
    ]
    context = _context_block(workdir, home or Path.home())
    if context is not None:
        parts.append(context)
    if registry.get("read") is not None:
        skills_block = format_skills_for_prompt(discover_skills(workdir))
        if skills_block:
            parts.append(skills_block)
    return "".join(parts)
