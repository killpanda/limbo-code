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

Limbo speaks to LLMs through a **provider/model catalog**
(`src/limbo/llm/catalog.py`). Each provider declares its API dialect,
endpoint, and credential env var; each model carries its own context window,
max output tokens, and thinking/reasoning behavior. A client factory
(`src/limbo/llm/factory.py`) picks the client implementation from the
provider's API dialect, so non-OpenAI dialects can be added without touching
call sites. Models not in the catalog fall back to generic
OpenAI-compatible defaults driven by `base_url` + `model` + `api_key`.

Built-in providers:

| Provider | API dialect | Endpoint | Key env var |
|----------|-------------|----------|-------------|
| `deepseek` | openai-completions | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `moonshotai` (Kimi) | openai-completions | `https://api.moonshot.ai/v1` | `MOONSHOT_API_KEY` |
| `kimi-coding` (Kimi For Coding) | anthropic-messages | `https://api.kimi.com/coding` | `KIMI_API_KEY` |
| `glm` (GLM Coding Plan) | openai-completions | `https://open.bigmodel.cn/api/coding/paas/v4` | `ZHIPUAI_API_KEY` |

Built-in GLM Coding Plan models: `glm-4.7`, `glm-5.1`, `glm-5.2` (1M
context), `glm-5-turbo`, `glm-5v-turbo` (vision), and `glm-4.5-air`. The
Coding Plan is a subscription with its own endpoint and keys — a coding-plan
key only works on the `/api/coding/paas/v4` endpoints (not the pay-per-token
`/api/paas/v4` ones) and vice versa. For the international endpoint set
`[providers.glm] base_url = "https://api.z.ai/api/coding/paas/v4"` (see
below).

Built-in Kimi models include `kimi-k3` (1M context, pay-per-token), and for
Kimi For Coding subscriptions: `k3` (1M context), `kimi-for-coding`, and
`kimi-for-coding-highspeed` (256K context). The two use different endpoints
and keys — a `sk-kimi-*` Kimi For Coding key only works with the
`kimi-coding` models (`k3`, ...) and vice versa.

Switching to a catalog model only requires changing `model` — the provider's
endpoint and key env var are picked up automatically:

```toml
[llm]
model = "k3"             # Kimi For Coding; base_url/api_key resolve from the catalog
```

The `anthropic-messages` dialect is served by a dedicated client
(`src/limbo/llm/anthropic_client.py`, plain httpx SSE) selected by the
client factory. It converts OpenAI-style tool definitions to Anthropic's
shape, merges consecutive tool results into a single user turn, and replays
assistant thinking blocks with their signature.

For the mainland-China Moonshot endpoint, set `base_url =
"https://api.moonshot.cn/v1"` explicitly (a configured `base_url` always
wins over the catalog — unless a `[providers.<id>]` override says otherwise,
see below).

### Per-provider overrides

An optional `[providers.<id>]` section overrides a single catalog provider
without touching the global `[llm]` settings — e.g. pointing a provider at a
relay, renaming its credential env var, or adding extra headers:

```toml
[providers.glm]
base_url = "https://api.z.ai/api/coding/paas/v4"  # international endpoint

[providers.codex]
base_url = "https://my-relay.example.com/v1"
api_key_env = "CODEX_API_KEY"     # rename the env var read for the key
headers = { x-relay = "on" }       # extra headers on every request
```

All fields are optional; unset fields fall back to the catalog. Resolution
order for `base_url` (first hit wins):

1. `[providers.<id>] base_url`
2. `[llm] base_url` (when changed from the DeepSeek default)
3. the catalog provider's built-in endpoint

and for the API key: `[providers.<id>] api_key` → `[llm] api_key` → the
environment variable (`[providers.<id>] api_key_env` rename, else the
catalog's default env var).

> **Note:** rule 1 is the single exception to the long-standing "an explicit
> `[llm] base_url` always wins" behavior — a per-provider override is more
> specific than the global setting. If you configure both, the
> `[providers.<id>]` value is used for that provider's models.

### Optional LLM settings

```toml
[llm]
temperature = 0.2        # 0.0 - 2.0, default 0.2
max_iterations = 50      # safety limit on tool-turn loops, default 50
max_tokens = 8192        # output cap; default = the model's catalog value
thinking_effort = "high" # reasoning control; default = provider behavior
```

`thinking_effort` is interpreted per model dialect:

- `k3`, `kimi-for-coding*` (Anthropic adaptive thinking): `low` | `high` |
  `max` → `thinking: {type: adaptive}` + `output_config.effort`. Thinking
  cannot be disabled; temperature is omitted while thinking is enabled.
- `kimi-k3` (moonshotai, OpenAI-style): `low` | `high` | `max` → sent as
  `reasoning_effort`. Thinking cannot be disabled on K3.
- `kimi-k2-thinking`, `kimi-k2.5`+ (DeepSeek-style): any non-`off` value →
  `thinking: {type: enabled}`; `"off"` → `thinking: {type: disabled}`
  (except `kimi-k2.7-code*`, where thinking is always on).
- `glm-*` (z.ai-style): like DeepSeek-style but with `clear_thinking: false`
  so thinking is preserved across turns; `glm-5.2` additionally maps
  `low`/`high`/`max` to `reasoning_effort` (`low` clamps to `high`).
- Non-reasoning models: ignored.

Reasoning output streams into the chat as muted thinking blocks and is
stored on the assistant message so it can be replayed to APIs that require
it (Kimi K3 rejects tool-call replays without `reasoning_content`).

Optional tool settings:

```toml
[tools]
bash_enabled = true
```

### Session storage

Conversations are saved as JSONL files in `~/.limbo/sessions/` so you can
review or debug them later. Use the `--session-dir` argument to redirect them
to another location.

Old session files are not automatically cleaned up; remove them manually when
you no longer need them.

## Run

```bash
limbo --workdir /path/to/project
```

## Safety

Limbo executes every tool call immediately, without asking for confirmation.
File tools (`read`, `edit`, `write`, `grep`, `find`, `ls`) are bounded to the
current working directory and reject paths that escape it, including via
symlinks. The boundary check resolves the path before each operation, so a
symlink swapped between the check and the operation (a time-of-check-to-time-of-use
race) could escape the workdir. **This is a known limitation for the MVP.**

Bash is an exception: it is started in the working directory but is **not**
sandboxed. Commands can `cd ..`, use absolute paths, and read or write outside
the workdir. In addition, commands that match dangerous patterns such as `rm`
or `git reset --hard` are **rejected outright**.
The pattern list is configurable but cannot be disabled from the UI. Bash
commands are filtered with a simple heuristic, but that filter can be bypassed
by subshells (`bash -c 'rm -rf /'`), command substitution (`$(rm -rf /)`),
variable indirection, options before the command name
(`git -C /foo reset --hard`), variable assignments before the command name
(`VAR=1 rm -rf /`), and similar shell constructs. Only run Limbo with
trusted commands and in repositories you can afford to modify or lose.

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
