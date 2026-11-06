# Limbo — Agent Guide

**Limbo** is a minimal terminal AI coding agent, written in Rust (edition 2024, ratatui + tachyonfx). Users converse with an LLM in a TUI to explore, read, edit, and write code via a small set of tools. `--headless` runs a plain-terminal REPL frontend instead of the TUI (Herdr/script control), `--json` prints one JSON event per line, and `-p "<prompt>"` runs one turn and exits.

**Tools execute immediately — no confirmation flow, no safety fences.** File tools are not scoped to the workdir (absolute paths and symlinks are honored as-is), there is no sensitive-file skip list, and `bash` runs every command as given (no dangerous-command filter). Every call passes admission (`admission.rs`): `ToolRegistry` runs its `Policy` list in order (`admit` returns `Deny(reason)` to block; `visible` hides a tool) — empty for the main agent, filled from the profile for subagents. `/plan` is the posture's policy dimension (`ToolRegistry.posture`, see `posture.rs`), which puts `PlanPolicy` first: only tools whose effect does not mutate run (bash is blocked), also enforced on Code Mode sub-calls, persisted as `SessionMeta.plan_mode`.

The source is commented for people learning Rust (`🦀 Rust notes` blocks); keep that style when adding code. `docs/learning-rust.md` has the reading order.

## Quick Start

```bash
cargo run --release -- --workdir /path/to/project   # config: ~/.limbo/config.toml
cargo install --path .                              # install the `limbo` binary
```

Runtime dependencies: `rg` (ripgrep) for `grep`; `python3` only for Code Mode (`run_code`) and the daily skills' helper scripts.

## Architecture

```
src/
├── main.rs           # CLI (clap): args, subcommands (models/auth/sessions/daily), launch
├── models.rs         # Message, ToolCall, ToolResult, LlmEvent (serde, the shared vocabulary)
├── config.rs         # ~/.limbo/config.toml: every section has defaults; toml_edit rewrites keep comments
├── auth.rs           # ~/.limbo/auth.json (0600)
├── cli_auth.rs       # `limbo auth …` / `limbo models`, first-run key prompt
├── llm/              # mod.rs (HTTP + retry + Decoder trait), catalog.rs (providers/models/Dialect),
│                     # openai.rs, anthropic.rs, responses.rs (the three dialects), sse.rs,
│                     # usage.rs (usage normalization + prompt-size estimation)
├── agent.rs          # The conversation loop (see below)
├── pump.rs           # Turn pump: one task = user turn + steer/follow-up drain + /compact + /goal loop
├── goal.rs           # /goal state machine, prompt templates, verify runner
├── compaction.rs     # Context compaction decision and summary prompt
├── history.rs        # tool_call↔result pairing + resume repair
├── admission.rs      # ActionRequest → Policy → Verdict (Allow/Deny); PlanPolicy
├── effect.rs         # Effect (read/fetch/session/write/mutate/exec) + Recovery (replay/inspect) per tool
├── attachments.rs    # Vision gate: images inline for vision models, path references otherwise
├── posture.rs        # Posture = policy (plan) × capability set (coding/daily) × presentation (direct/code); strategy (single/goal)
├── prompt.rs         # System-prompt assembly (tools section derived from the registry)
├── journal.rs        # Execution journal: `action` records (started/denied/completed/failed) + ActionLog for recovery
├── session.rs        # Session JSONL journal: records, append/rewrite, list, export, fork/handoff
├── trace.rs          # Append-only JSONL run log (<session dir>/traces/)
├── trace_viewer.rs   # /tracing-viewer: loopback HTTP server (std::net + threads) + assets/trace_viewer.html
├── commands.rs       # Frontend-neutral slash-command pieces (toggles, notices, /btw)
├── background.rs     # Process-wide channel for events outside a turn (subagents, skill reviews)
├── skills.rs         # SKILL.md discovery; the bundled daily pack (assets/daily_skills, include_dir!)
├── memory.rs         # ~/.limbo/memory/USER.md, NOTES.md: capped store + frozen per-session snapshot
├── learn.rs          # /learn and the automatic skill review trigger (daily mode)
├── recent_models.rs  # Recently used models for the picker
├── integrations.rs   # Herdr + OSC 7501 lifecycle reporting
├── headless.rs       # --headless / --json / -p frontends on the same pump
├── subagents/        # mod.rs (task board), supervisor.rs (scheduler, runners, notices),
│                     # tools.rs (plan_tasks/task_*/answer; child: checkpoint/need/ask_main/finish/propose_task),
│                     # profiles.rs (agent profiles + <project_facts> scan), policies.rs (read-only/no-edit/
│                     # allow-list/path claims), worktree.rs (git worktree isolation)
├── tools/            # mod.rs (Tool trait, ToolRegistry, path helpers, per-file mutation locks),
│                     # truncate.rs, read/write/edit/bash/grep/find/ls, run_code (Code Mode transport),
│                     # web_search (only when its provider has a key), daily mode only: web_fetch,
│                     # memory, session_search, ask_user (ends the turn), skill_manage
└── tui/              # mod.rs (event loop: tokio::select! over terminal/agent/background/ticks),
                      # app.rs (state, keys, slash commands), ui.rs (drawing), chat.rs (cached
                      # per-entry rendering), markdown.rs, text.rs (CJK-aware wrapping), input.rs,
                      # paste.rs, clipboard.rs, theme.rs, fx.rs (tachyonfx), game2048.rs
assets/
├── daily_skills/     # Bundled daily starter skills (ported from hermes-agent, MIT), embedded in the binary
└── trace_viewer.html # The /tracing-viewer page, embedded with include_str!
```

## Agent Loop

- The agent runs a turn and streams `AgentEvent`s (text/thinking deltas, tool call requests and results, progress, errors, compaction, usage, steer) over a channel to the frontend
- Loop top: auto-compaction check → steer drain → LLM call. A provider context-overflow error (413 or known overflow texts, never rate limits) triggers an overflow compaction and one resend per turn; a second overflow or a failed compaction surfaces as an error. The turn ends on a tool-call-free response; `max_iterations` (default 50) cancels pending calls with placeholder results
- Tool calls in one turn run **concurrently** (`[tools] parallel`); results are recorded in source order; same-file mutations are serialized by `tools::mutation_lock`
- `finish_reason` `length`/`max_tokens` → the whole batch fails without execution; the error names the effective `max_tokens` and steers the model to write large files via `write`. 3 consecutive length stops abort the turn
- Mid-turn user input queues as *steer*: injected at loop top, or as a turn-end follow-up. Alt+Enter (headless: `/queue <text>`) queues a `follow_up` instead, drained by the pump only after the whole task (goal loop included). Every queue change is journaled (`queue` records), so queued messages survive a crash and are restored on resume
- Reasoning is stored on assistant messages and replayed per dialect (Anthropic thinking signatures, Kimi `reasoning_content`, Responses encrypted items)
- Retryable failures before the first streamed event (429, 5xx, network) are retried inside the client with backoff; after the stream started, the request is resent up to `max_retries` times and the partial response is discarded
- Usage counters are normalized in `llm/usage.rs` and feed the compaction trigger; every billed call is added to the per-`provider/model` ledger `SessionMeta.usage`, so the token counter survives resume

## LLM & Config

Providers/models live in `llm/catalog.rs`; unknown models get generic OpenAI-compatible defaults. Resolution: `[providers.<id>]` overrides → explicit `[llm]` values → catalog defaults → `auth.json` / env-var API keys.

```toml
[llm]        # model, api_key, base_url, temperature, max_iterations,
             # max_tokens, thinking_effort, max_retries, timeout, ...
[tools]      # bash_enabled = true, parallel = true
[compaction] # enabled, reserve_tokens = 16384, keep_recent_tokens = 20000
[goal]       # max_rounds = 10, verify_timeout_ms
[queue]      # after_failed_run = "continue" | "keep" | "withdraw"
[subagents]  # enabled, auto = "off"|"suggest"|"on", max_parallel, max_tasks, ...
[daily]      # workdir = "~/limbo-daily", model, memory caps, learn trigger knobs, fetch limits
[web_search] # provider = "deepseek", api_key, count, enabled
[ui]         # theme (neon|aurora|ember|dawn|omarchy), motion (full|reduced|off)
[providers.<id>]  # base_url / api_key / api_key_env / headers
```

`/model` rebuilds the client mid-session (refused while busy) and persists via `toml_edit`. Sessions resume via `--continue` / `--resume` / `/sessions`; resume repairs dangling tool calls from the execution journal (`journal.rs`): each call appends an `action` record `started` (or `denied`) before it runs and `completed`/`failed` once its result is in (goal verify runs too, kind `verify`), so a crashed call is re-run on the next turn if its tool's recovery is `replay` (read/fetch effects: read/grep/find/ls, web tools), otherwise marked "ran to the end, result lost" (finished), "may have partially run" (started only) or "did not run" (nothing). Legacy `tool_intent` records load as `started`. `/export [path]` merges meta + trace + messages (Markdown if `.md`). `/fork [n]` copies the active history (first n user turns) into a new session, `/handoff <note>` starts a clean linked session from a note; both record `parent_session` and leave the parent untouched.

Skills: `~/.agents/skills/<name>/SKILL.md` (global, shared with other agents; lowest of the non-bundled), `~/.limbo/skills/<name>/SKILL.md` (user) and `<workdir>/.agents/skills/<name>/SKILL.md` (project, wins collisions); daily sessions also see the bundled pack (lowest, unpacked to `~/.limbo/daily/bundled/<version>/`) and `~/.limbo/daily/skills/` (highest), coding sessions never do — injected as an `<available_skills>` catalog in the system prompt and invocable via `/<name> [args]`.

## UI

Single column: status bar (state/elapsed/tokens/model/workdir/queue + goal and Code Mode badges) → scrolling transcript (user messages, streaming Markdown, tool cards, errors) → prompt box (Enter submits, Shift+Enter or Ctrl+J newline, Alt+Enter queues a follow-up, collapsed paste markers, Ctrl+V clipboard paste) → hint line. `/` opens the command palette; `@<agent> text` messages a subagent. Every color comes from a `Theme` in `tui/theme.rs`; widgets never hard-code colors.

`/goal <text>` starts closed-loop mode: the first round proposes acceptance command(s) in a `<verify_proposal>` block; the user confirms (accept / edit / skip), then each round is a full agent turn followed by the verify command. Exit 0 ends the loop; otherwise the failure output is fed back into the next round until `max_rounds`, when a no-tool wrap-up turn summarizes. Goal state persists on `SessionMeta.goal`. Esc during verify kills the subprocess.

Subagents (docs/subagents.md): the main agent's `plan_tasks` creates tasks on the board (suggest mode asks the user first); the supervisor runs each as a child agent with its own session file, gated by profile policies. Child tools that end the turn (`need`, `ask_main`, `finish`) return `end_turn`. Notices reach the main agent as one subagent steer item; an idle main agent is woken through the background channel. Board records live in the main session file (`type: "task"`), so the board survives restarts; tasks running at exit come back paused.

Daily mode (`limbo daily` / `--daily` / `/daily-mode on|off`, docs/daily-mode.md): a session posture like Code Mode — same loop, a general-assistant system prompt, plus the daily tools; persisted as `SessionMeta.daily_mode`. Memory is a snapshot taken when the agent is built: `USER.md` and `NOTES.md` enter daily prompts only; `memory` writes reach disk at once but the prompt changes only next session (prefix cache stays intact). `/learn` (and the automatic trigger in `learn.rs`) runs a `skill-reviewer` fork subagent whose only write tool is `skill_manage`.

## Key Decisions

- tokio everywhere; tools are `async_trait` objects (`Arc<dyn Tool>`), blocking work goes through `spawn_blocking` or child processes
- Code Mode (`/code-mode on|off`): `ToolRegistry.code_mode` collapses the catalog to `run_code`; `run_code` starts `python3` with a small bootstrap and forwards `tools.*` sub-calls back through the same registry over stdio. Only the program output (print + return) enters history. `SessionMeta.code_mode` persists the mode
- System prompt = hardcoded tool guidelines + `<workdir>/AGENTS.md` & `~/.limbo/AGENTS.md` (XML-wrapped) + skills catalog. **This file is injected into every request — keep it accurate and lean.**
- Session JSONL is checkpointed mid-turn: new messages are appended; full atomic rewrite (0600) only on first save, after a failed append, or when resume repaired dangling calls. Meta changes append `meta_patch`. Compaction is non-destructive: a `compaction` record is appended and loading selects the active range from the latest one. Unknown record types are skipped and unknown fields are preserved (`#[serde(flatten)]`). The trace holds the full-fidelity record

## Testing

```bash
cargo test                                  # unit tests live next to the code (#[cfg(test)] mod tests)
cargo fmt --check
cargo clippy --all-targets -- -D warnings   # CI treats warnings as errors
```

CI (`.github/workflows/ci.yml`) runs exactly these three.

## Adding Things

- **Tool**: implement `Tool` in `src/tools/<name>.rs` (`name`/`description`/`parameters` + `run()`; declare its `effect()` (`effect.rs`: read/fetch/session/write/mutate/exec — `replay_safe`/`mutates` derive from it; the default is `exec`), override `recovery`/`program_callable` when needed; resolve paths with `resolve_path`, take `mutation_lock(path)` around writes), add it to `ToolRegistry::core` or `set_daily`, and add tests in the same file
- **Web-search backend**: extend `src/tools/web_search.rs`; users pick it with `[web_search] provider`
- **Provider/model**: add specs in `llm/catalog.rs`; a new wire format → a new `Dialect` variant plus a module implementing `Decoder` and the request body. User-specific endpoint/key tweaks belong in `[providers.<id>]` config, not the catalog
