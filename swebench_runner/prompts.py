"""Prompt template for SWE-bench instances."""

from __future__ import annotations

TEMPLATE = """\
You are working on the repository `{repo}`, checked out at commit {base_commit}.
The repository is at /testbed (your current working directory), with all
dependencies pre-installed.

# Issue to resolve

{problem_statement}

# Instructions

- Diagnose the root cause, then modify the source code to resolve the issue.
- Keep the change minimal: fix the issue without refactoring unrelated code.
- Do NOT modify, add, or delete test files.
- Do NOT run `git commit`, `git add`, `git stash`, or `git checkout --`:
  your changes are collected via `git diff` after the run.
- You may run existing tests to verify your fix, but hidden tests will be
  used for the final evaluation.
- When you are confident the fix is complete, briefly summarize what you
  changed and why, then stop.
"""


def build_prompt(instance: dict) -> str:
    return TEMPLATE.format(
        repo=instance["repo"],
        base_commit=instance["base_commit"],
        problem_statement=instance["problem_statement"].strip(),
    )
