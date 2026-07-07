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
