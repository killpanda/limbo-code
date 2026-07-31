"""CLI entry point for Limbo."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from limbo.config import Config, load_config
from limbo.exec import run_headless
from limbo.llm.catalog import resolve_api_key, resolve_api_key_env, resolve_model
from limbo.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    find_session,
    latest_session,
    load_session,
)
from limbo.ui.app import LimboApp

DEFAULT_SESSION_DIR = Path.home() / ".limbo" / "sessions"
# Headless runs default to a separate dir so batch jobs (e.g. 500 SWE-bench
# instances) don't flood the interactive /sessions picker.
DEFAULT_EXEC_SESSION_DIR = Path.home() / ".limbo" / "exec-sessions"


def _check_api_key(config: Config) -> str | None:
    """Return an error message if no API key is configured, else None."""
    spec = resolve_model(config.llm.model)
    if resolve_api_key(spec, config):
        return None
    env = resolve_api_key_env(spec, config)
    env_hint = f" or ${env}" if env else ""
    return (
        "Error: No API key configured. Set it in ~/.limbo/config.toml\n"
        f"  [llm]\n  api_key = 'your-key'{env_hint}"
    )


def _exec_main(argv: list[str]) -> int:
    """`limbo exec`: one non-interactive agent turn (for batch drivers)."""
    parser = argparse.ArgumentParser(
        prog="limbo exec",
        description="Run one agent turn headlessly: prompt in, events out.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory (default: current directory)",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        default=None,
        help="Prompt text for the single turn",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Read the prompt from this file (preferred for long prompts)",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help=(
            "Directory for session JSONL files "
            f"(default: {DEFAULT_EXEC_SESSION_DIR})"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured model for this run (e.g. glm-4.7)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    if args.model:
        config.llm.model = args.model
    if error := _check_api_key(config):
        print(error, file=sys.stderr)
        return 1

    if args.prompt_file is not None:
        try:
            prompt = args.prompt_file.read_text(encoding="utf-8")
        except OSError as e:
            print(f"Error: Cannot read prompt file: {e}", file=sys.stderr)
            return 1
    else:
        prompt = args.prompt

    workdir = args.workdir.resolve()
    if not workdir.is_dir():
        print(f"Error: workdir does not exist: {workdir}", file=sys.stderr)
        return 1

    return asyncio.run(
        run_headless(
            prompt,
            config=config,
            workdir=workdir,
            session_dir=args.session_dir or DEFAULT_EXEC_SESSION_DIR,
        )
    )


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "exec":
        return _exec_main(sys.argv[2:])

    parser = argparse.ArgumentParser(description="Limbo TUI coding agent")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Directory for session JSONL files (default: ~/.limbo/sessions)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the most recent session for the working directory",
    )
    resume_group.add_argument(
        "--resume",
        metavar="SESSION_ID",
        default=None,
        help="Resume a session by id (or unique id prefix)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured model for this run (e.g. glm-4.7)",
    )
    args = parser.parse_args()

    config = load_config()
    if args.model:
        config.llm.model = args.model
    if error := _check_api_key(config):
        print(error, file=sys.stderr)
        return 1

    session_dir = args.session_dir or DEFAULT_SESSION_DIR
    workdir = args.workdir
    resume: Path | None = None

    if args.continue_session:
        resume = latest_session(session_dir, workdir=workdir)
        if resume is None:
            print(
                f"Error: No session to continue for {workdir.resolve()}",
                file=sys.stderr,
            )
            return 1
    elif args.resume is not None:
        try:
            resume = find_session(session_dir, args.resume)
        except SessionNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except AmbiguousSessionError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if resume is not None:
        # The session's recorded workdir wins when it still exists — sessions
        # are bound to the project they were created in.
        meta, _, _ = load_session(resume)
        if meta.workdir and Path(meta.workdir).is_dir():
            workdir = Path(meta.workdir)

    app = LimboApp(
        workdir=workdir,
        config=config,
        session_dir=args.session_dir,
        resume=resume,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
