"""CLI entry point for Limbo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from limbo.config import load_config
from limbo.ui.app import LimboApp


def main() -> int:
    parser = argparse.ArgumentParser(description="Limbo TUI coding agent")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory (default: current directory)",
    )
    args = parser.parse_args()

    config = load_config()
    if not config.llm.api_key:
        print(
            "Error: No API key configured. Set it in ~/.limbo/config.toml\n"
            "  [llm]\n  api_key = 'your-key'",
            file=sys.stderr,
        )
        return 1

    app = LimboApp(workdir=args.workdir, config=config)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
