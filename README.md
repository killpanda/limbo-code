# Limbo

A minimal terminal AI coding agent. Chat with an LLM in the terminal and it
explores, reads, edits, and runs your code — streaming its reasoning and every
tool call as it goes.

![Limbo](docs/assets/limbo-hero.png)

## Why

- **Small surface.** Seven tools — `read` `write` `edit` `bash` `grep` `find`
  `ls` — plus a clean agent loop. Nothing to configure before the first useful
  answer.
- **Any model.** A provider catalog covers DeepSeek, Kimi, GLM, and OpenAI
  Codex across three wire formats (OpenAI completions, Anthropic messages,
  OpenAI responses). Unknown models fall back to generic OpenAI-compatible
  defaults.
- **Fast turns.** Tool calls issued together run concurrently (same-file
  mutations are serialized), and **Code Mode** collapses the tool catalog to a
  single `run_code`: the model writes one Python program against a generated
  SDK, turning several round trips into one.
- **Work that survives.** Sessions are JSONL files you can resume, branch,
  hand off, or export; compaction keeps long runs alive without dropping the
  record.
- **No black boxes.** Full-fidelity traces of every request, tool call, and
  token count, browsable in a local viewer (`/tracing-viewer`).

## Install

Requires Python 3.11 or later.

```bash
pip install limbo-code
limbo auth login          # store a provider key in ~/.limbo/auth.json
limbo --workdir ~/code/my-project
```

Prefer a config file? `~/.limbo/config.toml`:

```toml
[llm]
api_key = "sk-..."                    # or just export DEEPSEEK_API_KEY
model = "deepseek-v4-pro"             # endpoint + key env resolve from the catalog
base_url = "https://api.deepseek.com/v1"
```

## Built-in providers

| Provider | Wire format | Endpoint |
|----------|-------------|----------|
| `deepseek` | openai-completions | `https://api.deepseek.com/v1` |
| `moonshotai` (Kimi) | openai-completions | `https://api.moonshot.ai/v1` |
| `kimi-coding` | anthropic-messages | `https://api.kimi.com/coding` |
| `glm` (GLM Coding Plan) | openai-completions | `https://open.bigmodel.cn/api/coding/paas/v4` |
| `codex` (OpenAI Codex) | openai-responses | `https://api.openai.com/v1` (relay expected) |

Switching models is a one-word change (`model = "k3"`) — or `-m glm-4.7` for a
single run, or `/model` mid-session. A `[providers.<id>]` section overrides one
provider without touching global settings: relay URL, renamed credential env
var, extra headers.

```toml
[providers.codex]
base_url = "https://my-relay.example.com/v1"
api_key_env = "CODEX_API_KEY"
headers = { x-relay = "on" }
```

## In the session

| Command | What it does |
|---------|--------------|
| `/model [name]` | switch model (picker when bare); persists to config, no restart |
| `/code-mode [on\|off]` | one `run_code` tool: a Python program, many tool calls |
| `/plan [on\|off]` | read-only mode — explore and propose, mutating tools blocked |
| `/goal <task>` | closed loop: propose a verify command, iterate until it passes |
| `/btw <question>` | side question that stays out of history |
| `/compact` | compact the conversation now |
| `/sessions` `/new` `/fork` `/handoff` | resume, start, branch, or hand off work |
| `/export [path]` | dump the session as JSONL or Markdown |
| `/tracing-viewer [id\|off]` | browse this run's trace in a local web page |

`Enter` sends, `Shift+Enter` newlines, `Alt+Enter` queues a follow-up for after
the current task, `Esc` interrupts.

Scripts and multiplexers get the same agent as a line-oriented REPL:
`limbo --headless` (or `--json` for one JSON event per line).

## Safety

**There are no fences.** Tools run immediately and unconfined: file tools honor
absolute paths and symlinks anywhere, there is no sensitive-file blocklist, and
`bash` runs every command as written — `rm -rf` included. Run Limbo with
trusted models in repositories you can afford to modify or lose.

```toml
[tools]
bash_enabled = false     # if you need a hard stop
```

## Development

```bash
uv sync        # dev tools are a PEP 735 group, installed by default
make check     # sync + tests + lint + typecheck
```

Individually — always `python -m pytest`, never `uv run pytest` (which can fall
through to a global pytest on `PATH`):

```bash
uv run python -m pytest tests/ -v
uv run ruff check src tests
uv run mypy src
```

The screenshot above is rendered from the real UI by
`scripts/readme_hero.py` (`uv run python scripts/readme_hero.py`), so it can't
drift from what ships.

## More

- `AGENTS.md` — architecture and design decisions
- `CONTEXT.md` — domain glossary
- `docs/` — session management, skills, UI design notes
