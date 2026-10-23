"""git worktree isolation for writer subagents (design §6.5).

A task with ``isolation = "worktree"`` runs in
``<workdir>/.limbo/worktrees/<task>`` on branch
``limbo/<session>/<task>``, created from the current HEAD. Its changes
are committed on that branch when it finishes; the main agent merges the
branch back with ``task_control(action="merge")``. Worktrees are kept
on disk (the board records the path) so a restart resumes in place.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorktreeError(Exception):
    pass


@dataclass(frozen=True)
class Worktree:
    path: Path
    branch: str


def _git(cwd: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise WorktreeError(str(e)) from e


def is_git_repo(workdir: Path) -> bool:
    try:
        return _git(workdir, "rev-parse", "--is-inside-work-tree").returncode == 0
    except WorktreeError:
        return False


def branch_name(session_id: str, task_id: str) -> str:
    return f"limbo/{session_id[-12:]}/{task_id}"


def _exclude_worktrees(workdir: Path) -> None:
    """Keep ``.limbo/`` out of the main checkout's ``git status``."""
    result = _git(workdir, "rev-parse", "--git-path", "info/exclude")
    if result.returncode != 0:
        return
    exclude = Path(result.stdout.strip())
    if not exclude.is_absolute():
        exclude = workdir / exclude
    try:
        current = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        if "/.limbo/" not in current.splitlines():
            exclude.parent.mkdir(parents=True, exist_ok=True)
            prefix = "" if not current or current.endswith("\n") else "\n"
            exclude.write_text(current + prefix + "/.limbo/\n", encoding="utf-8")
    except OSError:
        pass


def create(workdir: Path, session_id: str, task_id: str) -> Worktree:
    path = workdir / ".limbo" / "worktrees" / task_id
    branch = branch_name(session_id, task_id)
    if path.is_dir():
        return Worktree(path, branch)
    _exclude_worktrees(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = _git(workdir, "worktree", "add", "-b", branch, str(path), "HEAD")
    if result.returncode != 0:
        # The branch may survive from an earlier attempt: reuse it.
        result = _git(workdir, "worktree", "add", str(path), branch)
    if result.returncode != 0:
        raise WorktreeError(result.stderr.strip() or "git worktree add failed")
    return Worktree(path, branch)


def commit_all(path: Path, message: str) -> str:
    """Commit everything in the worktree; returns ``git diff --stat`` text."""
    _git(path, "add", "-A")
    stat = _git(path, "diff", "--cached", "--stat").stdout.strip()
    if stat:
        result = _git(
            path,
            "-c",
            "user.name=limbo",
            "-c",
            "user.email=limbo@localhost",
            "commit",
            "-m",
            message,
        )
        if result.returncode != 0:
            raise WorktreeError(result.stderr.strip() or "git commit failed")
    return stat


def merge(workdir: Path, branch: str) -> tuple[bool, str]:
    """Merge ``branch`` into the workdir's current branch (no fast-forward
    surprises: conflicts are aborted and reported)."""
    result = _git(workdir, "merge", "--no-edit", branch)
    if result.returncode == 0:
        return True, result.stdout.strip()
    _git(workdir, "merge", "--abort")
    return False, (result.stdout + result.stderr).strip()


def remove(workdir: Path, path: Path) -> None:
    _git(workdir, "worktree", "remove", "--force", str(path))
