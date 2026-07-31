# Limbo — Agent Guide

**Limbo** is a minimal terminal AI coding agent (Python 3.11+, Textual). Users converse with an LLM in a TUI to explore, read, edit, and write code via 7 tools.

**Tools execute immediately — no confirmation flow.** Convenience guardrails (not a security boundary): a workdir scope on file tools (+ session-scoped grants from user-mentioned paths), a sensitive-file skip list shared by `read`/`grep`/`find`, and a heuristic dangerous-command filter on `bash` (best-effort, rejects matches outright; `bash` is otherwise not covered by any guardrail).

## Quick Start

```bash
uv sync                               # installs runtime + dev tools (PEP 735 dev group, on by default)
uv run limbo --workdir /path/to/project   # config: ~/.limbo/config.toml
```

Prefer `uv` over `pip`: dev tools (pytest/ruff/mypy) are a `[dependency-groups]`
group, NOT an extra — `pip install -e ".[dev]"` no longer exists. `pip install -e .`
still works for a runtime-only install.

## Architecture

```
src/limbo/
├── app.py            # CLI entry: args, config, launch
├── config.py         # Pydantic config: llm/ui/safety/tools/compaction/providers
├── models.py         # Message, Attachment, ToolResult, LLMEvent
├── agent.py          # Conversation loop (see below)
├── attachments.py    # Attachment policy: vision gate, inline vs path-reference degrade
├── compaction.py     # Context compaction decision/prompt logic (LIM-14)
├── history.py        # tool_call↔result pairing + resume repair
├── model_switch.py   # /model domain logic: validate, swap client, persist (UI is an adapter)
├── prompt.py         # System-prompt assembly (tools section derived from the registry)
├── steer.py          # Mid-turn steer queue (LIM-20): queueing semantics, cancel boundary
├── goal.py           # /goal closed-loop state machine, prompt templates, verify executor (LIM-40)
├── goal_driver.py    # /goal orchestrator (frontend-agnostic, non-UI): turn → verify → next round
├── sessions.py       # Session JSONL save/load/list/export
├── trace.py          # Append-only JSONL run log (sessions/traces/)
├── skills.py         # SKILL.md discovery (user + project dirs)
├── user_paths.py     # Fence grants from paths in user messages
├── llm/              # catalog.py (provider/model specs), factory.py (dialect→client),
│                     # openai_client.py, anthropic_client.py, responses_client.py, retry.py, sse.py,
│                     # usage.py (token accounting: usage normalization + prompt-size estimation),
│                     # scaffold.py (plumbing shared by dialect clients: credentials, retry, images)
├── tools/            # base.py (BaseTool + path guardrail + truncation), registry.py (dispatch, grants),
│                     # mutation_queue.py (per-file locks), ignore.py (.gitignore),
│                     # read/bash/edit/write/grep/find/ls.py
└── ui/               # app.py + app.tcss (ALL styles here; theme vars only, no bare hex),
                      # theme.py (limbo-dark/-light, RFC LIM-16), commands.py (slash registry),
                      # screens/ (main, session_picker, model_picker, game2048),
                      # widgets/ (chat, input, status_bar, tool_card, command_menu)
```

## Agent Loop

- `Agent.run()` yields `AgentEvent`: `TextDelta`, `ThinkingDelta`, `ToolCallRequest`, `ToolResultEvent`, `ErrorEvent`, `CompactionEvent`, `UsageUpdate`, `SteerEvent`
- Loop top: auto-compaction check → steer drain → LLM call. Turn ends on a tool-call-free response; `max_iterations` (default 50) cancels pending calls with placeholder results
- Tool calls in one turn run **concurrently** (`[tools] parallel`); results stream in completion order, recorded in source order; same-file mutations serialized by `mutation_queue`
- `finish_reason` `length`/`max_tokens` → whole batch failed without execution; the error result names the effective `max_tokens` and steers the model to write large scripts via `write` instead of re-issuing the same call. 3 consecutive length stops abort the turn with actionable guidance (trace `kind=length_stop_loop`); a text-only truncated response emits a "may be incomplete" warning
- Mid-turn user input queues as *steer* (LIM-20): injected at loop top, or as turn-end follow-up
- Reasoning is stored on assistant messages and replayed per dialect (Anthropic thinking signatures, Kimi `reasoning_content`, Responses encrypted items)
- Usage counters normalize in `llm/usage.py` (`input_tokens`+`cache_read`, `prompt_tokens`, DeepSeek cache hits) and feed the compaction trigger via `PromptSizeEstimator`

## LLM & Config

Providers/models live in `llm/catalog.py`; unknown models get generic OpenAI-compatible defaults. Resolution: `[providers.<id>]` overrides → explicit `[llm]` values → catalog defaults → env-var API keys.

```toml
[llm]        # model, api_key, base_url, temperature, max_iterations,
             # max_tokens, thinking_effort, max_retries, timeout, ...
[tools]      # bash_enabled = true, parallel = true
[compaction] # enabled, reserve_tokens = 16384, keep_recent_tokens = 20000
[goal]       # max_rounds = 10, verify_timeout_ms = 600000
[safety]     # dangerous_commands, sensitive_files, auto_grant_user_paths
[ui]         # theme, show_banner
[providers.<id>]  # base_url / api_key / api_key_env / headers
```

`/model` rebuilds the client mid-session (refused while busy) and persists via tomlkit. Sessions resume via `--continue` / `--resume` / `/sessions`; resume repairs dangling tool calls and restores grants. Trace events: `session_start`, `user_message`, `llm_request` (full body), `llm_response`, `tool_call`, `tool_result`, `compaction*`, `error`, `turn_end`. `/export [path]` merges meta + trace + message snapshot (Markdown if `.md`).

Skills: `~/.limbo/skills/<name>/SKILL.md` (user) and `<workdir>/.agents/skills/<name>/SKILL.md` (project, wins collisions) — injected as a `<available_skills>` catalog in the system prompt and invocable via `/<name> [args]`.

## UI

Single column: status bar (state/elapsed/tokens/model/workdir/queued + rainbow 🎯GOAL badge while a goal loop runs) → scrolling chat flow (user `❯`, streaming Markdown, tool cards, errors) → input box (Enter submits, Shift+Enter newline, paste markers, `ctrl+v` image attach) → hint line. Slash commands: `/sessions /new /export /compact /model /help /2048 /goal` + skill commands. After palette changes run `python scripts/check_contrast.py` (≥ 4.5:1).

`/goal <text>` starts closed-loop mode (LIM-40): the first round is a proposal round — the model explores the repo and proposes acceptance command(s) in a `<verify_proposal>` block; the user confirms in a picker (accept / edit / skip), then `GoalDriver` (non-UI, shared by any frontend) runs each round as a complete `Agent.run()` turn and executes the verify command; exit 0 ends the loop, otherwise the failure output is fed back verbatim into the next round until `max_rounds`, when a no-tool wrap-up turn summarizes and any user message resumes with a fresh budget. Goal state persists on `SessionMeta.goal`. Esc during verify cancels the subprocess.

## Key Decisions

- Async agent, sync tools via `asyncio.to_thread()`
- System prompt = hardcoded tool guidelines + `<workdir>/AGENTS.md` & `~/.limbo/AGENTS.md` (XML-wrapped) + skills catalog. **This file is injected into every request — keep it accurate and lean.**
- Session files fully rewritten per save (atomic, 0600); trace holds full-fidelity record

## Testing

```bash
make check                          # single entrypoint: sync + tests + lint + type check
```

Or individually — always `python -m pytest`, never bare `uv run pytest` (which
silently falls through to a global pytest on PATH when the venv lacks pytest):

```bash
uv sync                             # first time / after pulling; dev group is default
uv run python -m pytest tests/ -v   # pytest-asyncio, respx HTTP mocks, pytest-textual-snapshot
uv run ruff check src tests && uv run mypy src  # lint + type check
uv run python -m pytest --snapshot-update  # after intentional visual changes; review the SVG diff
```

## Adding Things

- **Tool**: subclass `BaseTool` in `tools/<name>.py` (`name`/`description`/`parameters` + `run()`; use `resolve_existing()`/`resolve_creatable()`, raise `ToolError`), register in `ToolRegistry.__init__()`, wrap mutations in `mutation_lock_for(path)`, add `tests/tools/test_<name>.py`
- **Provider/model**: add specs in `llm/catalog.py`; new wire format → implement `LLMClient` + `register_client()` in `llm/factory.py`. User-specific endpoint/key tweaks belong in `[providers.<id>]` config, not the catalog
