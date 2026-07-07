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

### Provider

Limbo currently supports OpenAI-compatible chat completion endpoints. Set
`provider = "openai"` (the default) and configure `base_url`, `model`, and
`api_key` to point at your provider:

```toml
[llm]
provider = "openai"
base_url = "https://api.openai.com/v1"
model = "gpt-4o"
api_key = "your-api-key"
```

Optional tool settings:

```toml
[tools]
bash_enabled = true
```

### Session storage

Conversations are saved as JSONL files in `~/.limbo/sessions/` so you can
review or debug them later. Use the `--session-dir` argument to redirect them
to another location.

## Run

```bash
limbo --workdir /path/to/project
```

## Safety and confirmation

Limbo asks for confirmation before applying destructive or workspace-modifying
tool calls. Writes and edits are gated by default, and **bash is also
confirmation-gated** because it is unsandboxed and can mutate state.

File tools (`read`, `edit`, `write`, `grep`, `find`, `ls`) are bounded to the
current working directory and reject paths that escape it, including via
symlinks. The boundary check resolves the path before each operation, so a
symlink swapped between the check and the operation (a time-of-check-to-time-of-use
race) could escape the workdir. **This is a known limitation for the MVP.**

Bash is an exception: it is started in the working directory but is **not**
sandboxed. Commands can `cd ..`, use absolute paths, and read or write outside
the workdir. In addition, commands that match dangerous patterns such as `rm`
or `git reset --hard` are **rejected outright and cannot be confirmed**.
The pattern list is configurable but cannot be disabled from the UI. Bash
commands are filtered with a simple heuristic, but that filter can be bypassed
by subshells, command substitution, variable indirection, and similar shell
constructs. Only run Limbo with trusted commands and in repositories you can
afford to modify or lose.

If you need to work with untrusted projects, disable the bash tool entirely:

```toml
[tools]
bash_enabled = false
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
