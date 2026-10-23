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

from limbo.code_mode import CODE_ONLY_INSTRUCTION, render_tools_sdk
from limbo.memory import MemorySnapshot
from limbo.skills import discover_skills, format_skills_for_prompt
from limbo.tools.hooks import PLAN_MODE_INSTRUCTION

if TYPE_CHECKING:
    from limbo.tools.registry import ToolRegistry

_BASE_INTRO = (
    "You are an expert coding assistant operating inside limbo, "
    "a coding agent harness. You help users by reading files, "
    "executing commands, editing code, and writing new files.\n\n"
)

# Daily mode (non-coding posture): same harness, a general assistant voice.
_DAILY_INTRO = (
    "You are a capable personal assistant operating inside limbo. You help "
    "the user with everyday non-coding work: researching topics, reading web "
    "pages and documents, writing and editing text, organizing notes and "
    "files, and answering questions.\n\n"
)

_DAILY_GUIDELINES_TAIL = (
    "- For current facts, search (web_search) and read the sources (web_fetch) "
    "before answering; cite the URLs you relied on\n"
    "- Save substantial outputs (notes, reports, drafts) as files in the working "
    "directory and give their paths\n"
    "- Reply in the user's language; lead with the answer and keep it concise"
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
        if registry.get("write") is not None:
            # Pre-empt long heredoc one-liners: an inline script is a single
            # tool-call argument, so a big one can exceed the model's output
            # token limit mid-write (LIM-32).
            lines.append(
                "- Write multi-line scripts to a file with write, then run "
                "them with bash — avoid long heredoc one-liners"
            )
    if registry.daily_mode and registry.get("memory") is not None:
        lines.append(
            "- When the user tells you something durable about themselves or "
            "how they like things done, save it with memory"
        )
    if registry.daily_mode and registry.get("session_search") is not None:
        lines.append(
            "- When the user refers to earlier conversations, look them up with "
            "session_search instead of asking them to repeat"
        )
    if registry.daily_mode and registry.get("ask_user") is not None:
        lines.append(
            "- When a decision is the user's to make, ask with ask_user (offer "
            "options) instead of guessing"
        )
    if registry.daily_mode and registry.get("skill_manage") is not None:
        lines.append(
            "- When you work out a reusable way to do a kind of task, or the user "
            "corrects how a task should be done, record it in a skill with "
            "skill_manage (patch the skill you used before creating a new one)"
        )
    lines.append(_DAILY_GUIDELINES_TAIL if registry.daily_mode else _GUIDELINES_TAIL)
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
    memory: MemorySnapshot | None = None,
) -> str:
    """Assemble the system prompt: base text + tools + guidelines + context.

    The tools section and the bash guideline are derived from ``registry``,
    so the prompt always matches the tools the model can actually call.
    The skills catalog is included only when the read tool is available
    (the model loads SKILL.md files with read on demand). ``memory`` is the
    session's frozen memory snapshot, shown only in daily mode (coding
    sessions never see memory).
    """
    parts = [
        _DAILY_INTRO if registry.daily_mode else _BASE_INTRO,
        _tools_section(registry),
        "\n\n",
        _guidelines(registry),
    ]
    if registry.code_mode:
        # Code Mode (deepseek-harness parity): the catalog above is just
        # `run_code`; the SDK section tells the model how to reach every
        # other tool from inside a program.
        parts.append(
            f"\n\n{CODE_ONLY_INSTRUCTION}\n\n{render_tools_sdk(registry)}"
        )
    if registry.plan_mode:
        parts.append(f"\n\n{PLAN_MODE_INSTRUCTION}")
    if memory is not None and registry.daily_mode:
        block = memory.render()
        if block is not None:
            parts.append(block)
    context = _context_block(workdir, home or Path.home())
    if context is not None:
        parts.append(context)
    if registry.get("read") is not None:
        skills_block = format_skills_for_prompt(
            discover_skills(workdir, daily=registry.daily_mode)
        )
        if skills_block:
            parts.append(skills_block)
    return "".join(parts)
