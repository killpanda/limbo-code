"""CLI entry point for Limbo."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from limbo.config import load_config
from limbo.llm.catalog import (
    MAGPIE_PREFIX,
    missing_api_key_message,
    resolve_api_key,
    resolve_model,
)
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
    Chinese inside the herdr multiplexer turns into spaces under the full
    kitty keyboard protocol, and why herdr still needs the disambiguate flag.
    """
    mode = config.ui.kitty_keyboard
    if mode.disable_textual_kitty_key():
        os.environ["TEXTUAL_DISABLE_KITTY_KEY"] = "1"
    elif mode.disambiguate_only(in_herdr=bool(os.environ.get("HERDR_ENV"))):
        from limbo.ui.key_fixes import (  # noqa: PLC0415 - imports textual; must follow the env check above
            request_kitty_disambiguate_only,
        )

        request_kitty_disambiguate_only()


def _first_run_login(spec, config) -> bool:  # noqa: ANN001
    """No key anywhere and a real terminal: offer to log in instead of exiting."""
    from limbo.cli_auth import login  # noqa: PLC0415 - only on first run
    from limbo.llm.catalog import provider_display_name  # noqa: PLC0415

    name = provider_display_name(spec.provider.id)
    print(f"欢迎使用 limbo。还没有 {name} 的 API key（模型 {spec.id}）。")
    print("粘贴一个 key 就能开始，它会保存在 ~/.limbo/auth.json（仅自己可读）。")
    print("想换提供商：Ctrl+C 退出后运行 limbo auth login，或用 -m 指定模型。\n")
    try:
        if login(spec.provider.id) != 0:
            return False
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return resolve_api_key(spec, config) is not None


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("auth", "models"):
        from limbo.cli_auth import run  # noqa: PLC0415 - subcommands skip the TUI imports

        return run(sys.argv[1:])
    argv = sys.argv[1:]
    # `limbo daily ...` is `limbo --daily ...`.
    if argv[:1] == ["daily"]:
        argv = ["--daily", *argv[1:]]
    parser = argparse.ArgumentParser(description="Limbo TUI coding agent")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory (default: current directory; with --daily, "
        "[daily] workdir)",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Start in daily mode (non-coding assistant: web_fetch, memory) "
        "in the [daily] workdir; same as `limbo daily`",
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
        "-m",
        "--model",
        default=None,
        help="Override the configured model for this run (e.g. glm-4.7, "
        "kimi-k3:high to also set the thinking level)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the plain-terminal headless frontend instead of the TUI "
        "(line-oriented REPL for Herdr/script control; see "
        "design/rfc-headless-cli.md)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --headless (implied): print every event as one JSON object "
        "per line, so scripts can drive Limbo without parsing the screen",
    )
    args = parser.parse_args(argv)

    config = load_config()
    _apply_kitty_keyboard_workaround(config)
    if args.daily and config.daily.model and not args.model:
        config.llm.model = config.daily.model
    if args.model:
        from limbo.cli_auth import split_model_arg  # noqa: PLC0415

        config.llm.model, level = split_model_arg(args.model)
        if level:
            config.llm.thinking_effort = level
    if config.llm.model.startswith(MAGPIE_PREFIX):
        from limbo.llm import magpie  # noqa: PLC0415 - only for Magpie models

        magpie.load_for_startup(config.llm.model, config)
    spec = resolve_model(config.llm.model)
    if not resolve_api_key(spec, config) and not (
        sys.stdin.isatty() and sys.stdout.isatty() and _first_run_login(spec, config)
    ):
        print("Error: " + missing_api_key_message(spec, config), file=sys.stderr)
        return 1

    session_dir = args.session_dir or DEFAULT_SESSION_DIR
    workdir = args.workdir or Path.cwd()
    if args.daily and args.workdir is None:
        workdir = Path(config.daily.workdir).expanduser()
        try:
            workdir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error: cannot create daily workdir {workdir}: {e}", file=sys.stderr)
            return 1
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

    if args.headless or args.json:
        from limbo.headless import (
            run_headless,  # noqa: PLC0415 - deferred: keeps the TUI path's import cost off headless startup
        )

        return run_headless(
            workdir=workdir,
            config=config,
            session_dir=args.session_dir,
            resume=resume,
            json_mode=args.json,
            daily=args.daily,
        )

    from limbo.ui.app import LimboApp  # noqa: I001, PLC0415 - deferred so the kitty keyboard workaround runs first

    app = LimboApp(
        workdir=workdir,
        config=config,
        session_dir=args.session_dir,
        resume=resume,
        daily=args.daily,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
