# Single entrypoint for dev checks — humans and agents should only need
# `make check`. Dev tools (pytest/ruff/mypy) are a PEP 735 dependency group,
# so plain `uv sync` installs them.
#
# Note: tests run via `uv run python -m pytest`, NOT `uv run pytest` — the
# latter silently falls through to a global pytest on PATH when the venv
# lacks pytest, which can run the suite against the wrong source tree.

.PHONY: sync test lint typecheck check

sync:
	uv sync

test:
	uv run python -m pytest tests/ -v

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

check: sync test lint typecheck
