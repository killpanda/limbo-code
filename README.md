# Limbo

A minimal terminal AI coding agent.

## Install

Requires Python 3.11 or later.

```bash
pip install -e .
```

## Configure

Create `~/.limbo/config.toml`:

```toml
[llm]
api_key = "your-api-key"
model = "deepseek-chat"
base_url = "https://api.deepseek.com/v1"
```

Optional UI settings:

```toml
[ui]
confirm_writes = true
confirm_edits = true
```

## Run

```bash
limbo --workdir /path/to/project
```

## Safety and confirmation

Limbo asks for confirmation before applying destructive or workspace-modifying
tool calls (writes and edits by default). Bash commands are filtered with a
simple heuristic, but that filter can be bypassed by subshells, command
substitution, variable indirection, and similar shell constructs. Review every
command before confirming it, and only run Limbo in repositories you can afford
to modify or lose.

## Development

Run tests:

```bash
pytest tests/ -v
```

Run linting and type checks:

```bash
ruff check src tests
mypy src
```
