"""Project facts: a cheap, LLM-free look at the workdir (design §5.1).

The facts steer delegation: they are rendered into the main agent's
system prompt (``<project_facts>``), and ``/agents init`` turns them into
suggested project profiles (one worker per workspace package, a verifier
wired to the detected check command).
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from limbo.subagents.profiles import (
    PROJECT_AGENTS_DIR,
    Profile,
    SuggestedProfile,
    render_profile_md,
)

_LANGUAGES = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".go": "Go",
    ".rs": "Rust",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rb": "Ruby",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".h": "C/C++",
    ".cs": "C#",
    ".swift": "Swift",
    ".php": "PHP",
}
_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".limbo"}
_MAX_WALK = 20_000


@dataclass
class ProjectFacts:
    is_git: bool = False
    dirty: bool = False
    file_count: int = 0
    languages: list[tuple[str, int]] = field(default_factory=list)
    packages: list[str] = field(default_factory=list)
    check_commands: list[str] = field(default_factory=list)
    has_agents_md: bool = False
    project_profiles: list[str] = field(default_factory=list)

    @property
    def large(self) -> bool:
        return self.file_count >= 1500


def _git(workdir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _list_files(workdir: Path) -> tuple[list[str], bool]:
    listed = _git(workdir, "ls-files")
    if listed is not None:
        return [line for line in listed.splitlines() if line], True
    files: list[str] = []
    for path in workdir.rglob("*"):
        if len(files) >= _MAX_WALK:
            break
        if any(part in _SKIP_DIRS for part in path.relative_to(workdir).parts):
            continue
        if path.is_file():
            files.append(str(path.relative_to(workdir)))
    return files, False


def _packages(workdir: Path) -> list[str]:
    found: list[str] = []
    pkg_json = workdir / "package.json"
    if pkg_json.is_file():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        workspaces = data.get("workspaces") if isinstance(data, dict) else None
        if isinstance(workspaces, dict):
            workspaces = workspaces.get("packages")
        for pattern in workspaces or []:
            if isinstance(pattern, str):
                found += [
                    str(p.relative_to(workdir))
                    for p in sorted(workdir.glob(pattern))
                    if (p / "package.json").is_file()
                ]
    pyproject = workdir / "pyproject.toml"
    if pyproject.is_file():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        members = data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
        for pattern in members:
            found += [
                str(p.relative_to(workdir)) for p in sorted(workdir.glob(pattern)) if p.is_dir()
            ]
    go_work = workdir / "go.work"
    if go_work.is_file():
        try:
            text = go_work.read_text(encoding="utf-8")
        except OSError:
            text = ""
        found += re.findall(r"^\s*(?:use\s+)?\.\/(\S+)", text, re.MULTILINE)
    cargo = workdir / "Cargo.toml"
    if cargo.is_file():
        try:
            data = tomllib.loads(cargo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for pattern in data.get("workspace", {}).get("members", []):
            found += [
                str(p.relative_to(workdir)) for p in sorted(workdir.glob(pattern)) if p.is_dir()
            ]
    return list(dict.fromkeys(found))


def _check_commands(workdir: Path) -> list[str]:
    commands: list[str] = []
    makefile = workdir / "Makefile"
    if makefile.is_file():
        try:
            targets = re.findall(
                r"^([A-Za-z][\w-]*):", makefile.read_text(encoding="utf-8"), re.MULTILINE
            )
        except OSError:
            targets = []
        for target in ("check", "test", "lint"):
            if target in targets:
                commands.append(f"make {target}")
                break
    pkg_json = workdir / "package.json"
    if pkg_json.is_file():
        try:
            scripts = json.loads(pkg_json.read_text(encoding="utf-8")).get("scripts", {})
        except (OSError, ValueError, AttributeError):
            scripts = {}
        for name in ("check", "test"):
            if isinstance(scripts, dict) and name in scripts:
                commands.append(f"npm run {name}")
                break
    if (workdir / "pyproject.toml").is_file() and not commands:
        commands.append("python -m pytest")
    if (workdir / "go.mod").is_file():
        commands.append("go test ./...")
    if (workdir / "Cargo.toml").is_file():
        commands.append("cargo test")
    return commands


def scan_project(workdir: Path) -> ProjectFacts:
    """Collect the facts (tens of milliseconds on a typical repo)."""
    files, is_git = _list_files(workdir)
    counts: Counter[str] = Counter()
    for name in files:
        lang = _LANGUAGES.get(Path(name).suffix.lower())
        if lang:
            counts[lang] += 1
    status = _git(workdir, "status", "--porcelain") if is_git else None
    profiles_dir = workdir / PROJECT_AGENTS_DIR
    return ProjectFacts(
        is_git=is_git,
        dirty=bool(status and status.strip()),
        file_count=len(files),
        languages=counts.most_common(5),
        packages=_packages(workdir),
        check_commands=_check_commands(workdir),
        has_agents_md=(workdir / "AGENTS.md").is_file(),
        project_profiles=sorted(p.stem for p in profiles_dir.glob("*.md"))
        if profiles_dir.is_dir()
        else [],
    )


def format_facts(facts: ProjectFacts) -> str:
    lines = ["<project_facts>"]
    langs = ", ".join(f"{name} {n}" for name, n in facts.languages) or "未识别"
    lines.append(
        f"- 文件数 {facts.file_count}（{'大' if facts.large else '中小'}型仓库）；语言：{langs}"
    )
    if facts.packages:
        lines.append(f"- 工作区子包（天然的并行切分单位）：{', '.join(facts.packages[:20])}")
    if facts.check_commands:
        lines.append(f"- 检测到的验证命令：{', '.join(facts.check_commands)}")
    if facts.is_git:
        lines.append(f"- git 仓库{'，有未提交改动' if facts.dirty else '，工作区干净'}")
    else:
        lines.append("- 不是 git 仓库（worktree 隔离不可用）")
    lines.append("</project_facts>")
    return "\n".join(lines)


def suggest_profiles(facts: ProjectFacts, workdir: Path) -> list[SuggestedProfile]:
    """Project profiles ``/agents init`` proposes (never written unasked)."""
    target = workdir / PROJECT_AGENTS_DIR
    suggestions: list[SuggestedProfile] = []
    for package in facts.packages[:8]:
        slug = re.sub(r"[^a-z0-9-]+", "-", package.lower()).strip("-") or "pkg"
        profile = Profile(
            name=f"{slug}-worker",
            description=f"只改 {package}/ 下的代码",
            when_to_use=f"任务只涉及 {package}/ 时",
            tools="all",
            scope_paths=(f"{package}/**",),
            body=(
                f"You are the subagent for the `{package}` package. Stay inside "
                f"`{package}/`; follow its own README/AGENTS.md if present."
            ),
        )
        suggestions.append(
            SuggestedProfile(
                profile.name,
                target / f"{profile.name}.md",
                render_profile_md(profile),
                reason=f"工作区子包 {package}",
            )
        )
    if facts.check_commands:
        command = facts.check_commands[0]
        profile = Profile(
            name="project-verifier",
            description=f"运行 `{command}` 并定位失败",
            when_to_use="改动完成后做整体验证",
            tools="no_edit",
            body=(
                f"Run `{command}` from the project root. Do not edit source "
                "files. Report pass/fail and, for each failure, the smallest "
                "actionable diagnosis."
            ),
        )
        suggestions.append(
            SuggestedProfile(
                profile.name,
                target / f"{profile.name}.md",
                render_profile_md(profile),
                reason=f"检测到验证命令 {command}",
            )
        )
    return [s for s in suggestions if not s.path.exists()]
