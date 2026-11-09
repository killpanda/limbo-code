# Limbo

**A terminal AI coding agent that is small enough to understand and sturdy
enough to trust with long work.**

Tell Limbo what you want in plain language. It explores your repository, reads
and edits files, runs commands, and streams its reasoning and every tool call
as it goes. When the work runs long, Limbo keeps it safe: every turn is
journaled to disk, a crash or a closed terminal picks up where it stopped, and
you can branch, hand off, or replay any session.

## Who it's for

- **Developers who live in the terminal** and want an agent that fits next to
  their editor, multiplexer, and dotfiles instead of a separate app.
- **People who use more than one model.** DeepSeek, Kimi, GLM, OpenAI Codex, or
  any OpenAI-compatible endpoint, switchable mid-conversation.
- **Anyone who wants to see what the agent did.** Every request, tool call, and
  token is recorded and browsable; nothing happens off-screen.
- **Hackers who want to read the code.** A small Rust crate: a handful of
  tools, one agent loop, plain JSONL on disk. `AGENTS.md` explains the design,
  and the source is commented for people learning Rust.

## What you get

### Work that survives

- **Durable sessions.** Messages, tool-call intents, queued follow-ups, and
  compactions are appended to a JSONL journal as they happen. Kill the process
  mid-turn and resume: read-only calls that were interrupted are re-run,
  calls that may have had side effects are flagged so the model checks before
  retrying, and nothing you queued is lost.
- **Resume, branch, hand off.** `limbo --continue` or `/sessions` to pick up,
  `/fork` to try another direction without touching the original, `/handoff`
  to start a clean context with a written brief that links back.
- **Long runs stay alive.** Automatic compaction summarizes old turns when the
  context fills up, while the full record stays in the journal.

### You stay in control

- **`/plan`** puts the agent in read-only mode: it explores and proposes a
  step-by-step plan, while writes, edits, and mutating shell commands are
  blocked until you say `/plan off`.
- **`/goal <task>`** closes the loop: the model proposes a verify command, you
  confirm it, and Limbo iterates until it passes.
- **Steer while it works.** Type while a turn is running to redirect it;
  `Alt+Enter` queues a follow-up for after the task; `Esc` interrupts.
- **Background subagents.** The agent can split work into tasks that run in
  parallel with their own context: they checkpoint, wait on each other when
  they discover a dependency, and report back. You approve the plan, watch
  the board in `/agents`, and pause, cancel or talk to any of them with
  `/task` or `@name`, or just ask the main agent.
- **Side channels.** `/btw <question>` asks something quick without polluting
  the history; `!<command>` runs a shell command yourself, never sent to the
  model.

### Any model, set up in seconds

- **Guided setup.** On first run with no key, Limbo asks for one and saves it
  to `~/.limbo/auth.json` (readable only by you). `limbo auth login` verifies a
  key against the provider before saving; `limbo models` lists the catalog and
  which models are ready to use.
- **Switch without restarting.** `/model` opens a searchable picker with
  recent models first. Pick a model whose provider has no key yet
  and it asks for one inline. `Shift+Tab` steps through thinking levels.
- **Three wire formats** (OpenAI completions, Anthropic messages, OpenAI
  responses) behind one catalog, with per-provider overrides for relays,
  renamed key variables, and extra headers.

### Fast turns

- Tool calls issued together run concurrently; edits to the same file are
  serialized.
- **Code Mode** (`/code-mode`) collapses the tool catalog into a single
  `run_code` tool: the model writes one Python program against a generated SDK,
  turning many round trips into one.

### Nothing hidden

- **`/tracing-viewer`** opens a local web page with a live view of every LLM
  request, tool call, and token count in the session.
- **`/export`** dumps the session as a full JSONL log (request bodies, usage,
  tool results, errors) or as readable Markdown.

### A TUI worth living in

- Five themes: `neon` (default), `aurora`, `ember`, `dawn` (the daily-mode
  default) and `omarchy`, which follows the Omarchy desktop theme. Switch with
  `/theme` or `[ui] theme = "aurora"`.
- **Motion.** Subtle animation built on tachyonfx. Set
  `[ui] motion = "reduced"` or `"off"` if you prefer stillness (SSH sessions
  default to reduced).
- `/` opens the command palette, `Ctrl+V` pastes images, file paths or text,
  `Ctrl+Y` copies the last reply, `Ctrl+O` expands or collapses tool output.

### Daily mode for everything else

`limbo daily` (or `/daily-mode on`) turns Limbo into an everyday assistant for
non-coding work, in its own working directory:

- **Memory.** `USER.md` (who you are) and `NOTES.md` (what it learned) under
  `~/.limbo/memory/`, loaded into daily sessions only, never into coding ones.
- **More tools.** `web_fetch` reads pages and PDFs, `session_search` recalls
  past conversations, and `ask_user` asks you a multiple-choice question.
- **A starter pack of skills** for PDFs, Word, Excel, slides, citations,
  meeting notes, weekly reviews, YouTube and arXiv. Daily mode has its own
  skill directory (`~/.limbo/daily/skills/`) that coding sessions never see.
- **It learns.** `/learn`, or an automatic review after a long stretch of
  work, has a background subagent write what it learned into skills. Every
  write lands in the daily skill directory and keeps a `.bak`.

See [docs/daily-mode.md](docs/daily-mode.md).

### From Telegram and WeChat

```bash
limbo channels login telegram --token <BotFather token>   # then message the bot once
limbo channels login weixin                               # scan the QR code with WeChat
limbo gateway install                                     # launchd on macOS, systemd on Linux
```

Each chat gets its own daily-mode session; only the owners recorded at
login can talk to it. See [docs/gateway.md](docs/gateway.md).

### Scriptable

`limbo --headless` gives the same agent as a line-oriented REPL, and
`limbo --json` prints one JSON event per line so scripts can drive it without
parsing the screen. Inside the [Herdr](https://herdr.dev) multiplexer, Limbo
reports its working, idle, and blocked state to the pane.

## Quick start

Requires Rust 1.88 or later (edition 2024) and `rg` (ripgrep) on `PATH`.
Code Mode and the daily skills' helper scripts also need `python3`.

```bash
git clone https://github.com/killpanda/limbo-code && cd limbo-code
cargo install --path .      # installs the `limbo` binary into ~/.cargo/bin
cd ~/code/my-project
limbo                       # first run asks for an API key
```

Prefer to set up keys first:

```bash
limbo auth login            # choose a provider, paste a key, verified and saved
limbo auth list             # where each provider's key comes from
limbo models                # what you can run right now
limbo -m glm-4.7            # one run on a different model
limbo -m kimi-k3:high       # ...with a thinking level
```

Then just describe the task: *"find why the login test is flaky and fix it"*.

## Everyday use

| Key | Action |
|-----|--------|
| `Enter` / `Shift+Enter` (or `Ctrl+J`) | send / new line |
| `Enter` while the agent works | steer: the message is delivered mid-turn |
| `Alt+Enter` | queue a follow-up for after the current task |
| `Esc`, `Ctrl+C` | interrupt; `Ctrl+C` twice when idle quits (with text selected, `Ctrl+C` copies instead) |
| `Super+C` / `Ctrl+Shift+C` | copy the mouse selection (`Super+V` / `Ctrl+Shift+V` paste) |
| `Shift+Tab` | cycle thinking level |
| `/` | command palette |
| `Ctrl+R` | sessions |
| `Ctrl+G` | subagent panel |
| `Ctrl+O` / `Ctrl+T` | expand or collapse tool output / show or hide thinking |
| `Ctrl+V` / `Ctrl+Y` | paste from the clipboard / copy the last reply |
| `PageUp` / `PageDown`, `Ctrl+↑/↓` | scroll the conversation |

| Command | What it does |
|---------|--------------|
| `/model [name]` | switch model (picker when bare); persists to config |
| `/plan [on\|off]` | read-only mode: explore and propose, mutations blocked |
| `/goal <task>` | iterate until a confirmed verify command passes |
| `/code-mode [on\|off]` | one `run_code` tool: a Python program, many tool calls |
| `/btw <question>` | side question that stays out of history |
| `/agents` `/task` `@name` | background subagents: the task board, direct control, talk to one |
| `/compact` | compact the conversation now |
| `/sessions` `/new` | switch to a past session, or start fresh |
| `/fork [n]` | branch into a new session (optionally keep only the first n turns) |
| `/handoff <brief>` | clean context seeded with a brief, linked to this session |
| `/export [path]` | full JSONL log, or Markdown when the path ends in `.md` |
| `/tracing-viewer [id\|off]` | live trace of this session in your browser |
| `!<command>` | run a shell command yourself, outside the conversation |

From the shell: `limbo --continue` (`-c`) resumes the latest session for the
current directory, `limbo --resume <id-prefix>` (`-r`) resumes any session,
`limbo sessions [--all]` lists them, and `limbo -p "<prompt>"` runs one turn
and prints the answer.

## Models and providers

| Provider | Wire format | Endpoint |
|----------|-------------|----------|
| `deepseek` | openai-completions | `https://api.deepseek.com/v1` |
| `moonshotai` (Kimi) | openai-completions | `https://api.moonshot.ai/v1` |
| `kimi-coding` | anthropic-messages | `https://api.kimi.com/coding` |
| `glm` (GLM Coding Plan) | openai-completions | `https://open.bigmodel.cn/api/coding/paas/v4` |
| `codex` (OpenAI Codex) | openai-responses | `https://api.openai.com/v1` (relay expected) |
| `magpie` ([Magpie](https://usemagpie.ai/) local gateway) | openai-completions | `http://127.0.0.1:3425/v1` |

Magpie models are addressed as `magpie/<upstream>/<model>` (e.g.
`limbo -m magpie/deepseek/deepseek-flash`, or a routing group such as
`magpie/auto/deepseek-v4-pro`); limbo sends the part after `magpie/` to the
gateway. The model list is read from the running gateway's `/v1/models`
(`limbo models`, the `/model` picker, and startup with a Magpie model), and
so are each model's context window, output cap, image input, reasoning
levels and wire format (models that only serve Responses or Messages are
spoken to natively). No key is needed: the placeholder `magpie` is sent
unless `$MAGPIE_API_KEY` or `limbo auth login magpie` sets one.

Unknown model ids fall back to generic OpenAI-compatible defaults, and the
model picker can connect any OpenAI-compatible endpoint. Everything can also
live in `~/.limbo/config.toml`:

```toml
[llm]
model = "deepseek-v4-pro"           # endpoint and key variable come from the catalog

[providers.codex]                   # override one provider only
base_url = "https://my-relay.example.com/v1"
api_key_env = "CODEX_API_KEY"
headers = { x-relay = "on" }

[ui]
theme = "neon"                      # neon | aurora | ember | dawn | omarchy
motion = "full"                     # full | reduced | off
```

### Web search

The `web_search` tool is offered whenever its provider has a key. The only
provider today is DeepSeek
([search API](https://api-docs.deepseek.com/guides/search)); without its own
key it reuses the DeepSeek model key (`[providers.deepseek]`, `limbo auth
login`, `$DEEPSEEK_API_KEY`, or `[llm] api_key` while a DeepSeek model is
active).

```toml
[web_search]
provider = "deepseek"    # the only backend today (src/tools/web_search.rs)
api_key = "sk-..."       # optional: a separate key just for search
count = 10               # default results per search
enabled = true
```

## Make it yours

- **Project instructions.** Limbo reads `AGENTS.md` from the working directory
  and `~/.limbo/AGENTS.md` into its system prompt.
- **Skills.** Drop a `SKILL.md` into `<project>/.agents/skills/<name>/`,
  `~/.limbo/skills/<name>/` or the cross-agent `~/.agents/skills/<name>/`. The model loads it when the task matches, or you
  call it as `/<name>`. See [docs/skills.md](docs/skills.md).

## Safety

**There are no fences.** Tools run immediately and unconfined: file tools
honor absolute paths and symlinks anywhere, there is no sensitive-file
blocklist, and `bash` runs every command as written, `rm -rf` included. Use
`/plan` when you only want the agent to look, run Limbo with trusted models,
and keep it in repositories you can afford to modify or lose.

```toml
[tools]
bash_enabled = false     # if you need a hard stop
```

## Development

```bash
cargo run --release         # run from the checkout (add -- <args> for flags)
cargo test                  # tests
cargo fmt --check           # formatting
cargo clippy --all-targets -- -D warnings
```

CI runs exactly those three checks. The crate layout and a suggested reading
order (with notes on the Rust ideas each module shows off) are in
[docs/learning-rust.md](docs/learning-rust.md).

## Acknowledgements

Limbo borrows openly from other agents. Thank you to:

- **[pi](https://github.com/badlogic/pi-mono) and pi-durable.** The durable
  session design comes from pi-durable: the "commit the intent, then the
  effect" journal, the replay policy for tool calls interrupted by a crash,
  the placeholders that say what is known about an unfinished call, the
  steer and follow-up inbox (`Alt+Enter`), the task graph edges behind
  background subagents, and context overflow detection. From the pi coding
  agent itself: the built-in tool behavior (output limits, multi-edit,
  process-tree kill, ripgrep), the provider conventions and model catalog,
  compaction and the session entry model, and collapsing large pastes.
- **[Crush](https://github.com/charmbracelet/crush)** by Charm. The Limbo
  TUI was designed after Crush: the gradient wordmark, `╱` dialog rules,
  the staggered entrance motion, the command palette and the session picker.
- **[Hermes Agent](https://github.com/NousResearch/hermes-agent)** by Nous
  Research. Daily mode is modeled on Hermes: the `USER.md` / `NOTES.md`
  memory with its frozen per-session snapshot and the memory nudge,
  `session_search`, `ask_user`, and skill self-learning (the skill reviewer's
  prompt is adapted from Hermes). The bundled daily skills in
  `assets/daily_skills/` are ported from Hermes under the MIT license, and
  each keeps its `LICENSE`.

## Learn more

- [`AGENTS.md`](AGENTS.md): architecture and design decisions
- [`CONTEXT.md`](CONTEXT.md): domain glossary
- [`docs/session-management.md`](docs/session-management.md): sessions,
  recovery, fork and handoff, tracing viewer
- [`docs/skills.md`](docs/skills.md): writing skills
- [`docs/daily-mode.md`](docs/daily-mode.md): daily mode, memory and
  self-learning skills
- [`docs/subagents.md`](docs/subagents.md): background subagents, the task
  board and how to steer them
- [`docs/gateway.md`](docs/gateway.md): talk to limbo from Telegram and
  WeChat; run it as a macOS launchd or Linux systemd service
- [`docs/learning-rust.md`](docs/learning-rust.md): code map and reading
  order for learning Rust from this codebase
- [`design/`](design/): RFCs for headless mode, bang commands, and the UI
