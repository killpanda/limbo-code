"""CLI entry point for Limbo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from limbo.config import load_config
from limbo.llm.catalog import resolve_api_key, resolve_api_key_env, resolve_model
from limbo.sessions import (
    AmbiguousSessionError,
    SessionNotFoundError,
    find_session,
    latest_session,
    load_session,
)

DEFAULT_SESSION_DIR = Path.home() / ".limbo" / "sessions"


def _apply_kitty_keyboard_workaround(config) -> None:
    """Apply the `ui.kitty_keyboard` setting before Textual initializes.

    Textual reads TEXTUAL_DISABLE_KITTY_KEY at import time, so this must run
    before the first `textual` import. See KittyKeyboardMode for why typing
    Chinese inside the herdr multiplexer turns into spaces when the kitty
    keyboard protocol is enabled.
    """
    if config.ui.kitty_keyboard.disable_textual_kitty_key(
        in_herdr=bool(os.environ.get("HERDR_ENV"))
    ):
        os.environ["TEXTUAL_DISABLE_KITTY_KEY"] = "1"


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
    parser.add_argument(
        "--model",
        default=None,
        help="Override the configured model for this run (e.g. glm-4.7)",
    )
    args = parser.parse_args()

    config = load_config()
    _apply_kitty_keyboard_workaround(config)
    if args.model:
        config.llm.model = args.model
    spec = resolve_model(config.llm.model)
    if not resolve_api_key(spec, config):
        env = resolve_api_key_env(spec, config)
        env_hint = f" or ${env}" if env else ""
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

    from limbo.ui.app import LimboApp  # noqa: I001, PLC0415 - deferred so the kitty keyboard workaround runs first

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
