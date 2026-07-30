"""CLI entry point for Limbo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from limbo.config import load_config
from limbo.llm.catalog import resolve_api_key, resolve_model
from limbo.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    find_session,
    latest_session,
    load_session,
)
from limbo.ui.app import LimboApp

DEFAULT_SESSION_DIR = Path.home() / ".limbo" / "sessions"


def main() -> int:
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
    args = parser.parse_args()

    config = load_config()
    spec = resolve_model(config.llm.model)
    if not resolve_api_key(spec, config):
        env_hint = (
            f" or ${spec.provider.api_key_env}"
            if spec.provider.api_key_env
            else ""
        )
        print(
            "Error: No API key configured. Set it in ~/.limbo/config.toml\n"
            f"  [llm]\n  api_key = 'your-key'{env_hint}",
            file=sys.stderr,
        )
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
