# Limbo — Agent Guide

## Overview

**Limbo** is a minimal terminal AI coding agent built with Python 3.11+ and Textual. It provides a TUI where users converse with an LLM (OpenAI-compatible, defaulting to DeepSeek) to explore, read, edit, and write code using a set of 7 tools. All destructive operations (`write`, `edit`, `bash`) require explicit user confirmation in the TUI.

## Quick Start

```bash
pip install -e .            # install
limbo --workdir /path/to/project   # run
```

Configuration lives in `~/.limbo/config.toml` (see [README](./README.md)).

## Architecture

```
src/limbo/
├── app.py                  # CLI entry point: arg parsing, config loading, app launch
├── __init__.py             # Version string
├── __main__.py             # `python -m limbo` support
├── config.py               # Pydantic models for config (LLM, UI, safety, tools)
├── models.py               # Shared data types: Message, ToolCall, ToolResult, LLMEvent
├── agent.py                # Conversation loop: orchestrates LLM + tools, handles confirmation
├── history.py              # ToolHistory: tool_call ↔ tool-result pairing bookkeeping + resume repair
├── sessions.py             # Session storage: save/load/list/find/export (meta line + JSONL messages)
├── trace.py                # TraceLogger: append-only JSONL run log (traces/ subdir) for full-fidelity analysis
├── skills.py               # Skill discovery: scan SKILL.md dirs (user + project), parse frontmatter
├── llm/
│   ├── client.py           # LLMClient Protocol
│   └── openai_client.py    # OpenAI-compatible streaming client
├── tools/
│   ├── base.py             # BaseTool (execute/run template) + path resolution + confirm stub + truncation
│   ├── ignore.py           # GitignoreMatcher: shared root .gitignore engine for find/grep
│   ├── registry.py         # ToolRegistry: collects and dispatches tools
│   ├── read.py             # Read file contents
│   ├── bash.py             # Execute bash commands (with safety filter)
│   ├── edit.py             # Surgical text replacement in files
│   ├── write.py            # Create/overwrite files
│   ├── grep.py             # Search file contents (ripgrep or Python fallback)
│   ├── find.py             # Find files by glob
│   └── ls.py               # List directory contents
└── ui/
    ├── app.py              # Textual App subclass (CSS_PATH = app.tcss)
    ├── app.tcss            # Centralized stylesheet for the whole TUI
    ├── theme.py            # limbo-dark / limbo-light Theme definitions (RFC LIM-16 palette)
    ├── syntax.py           # Pygments styles matching the theme palette (tool-card highlighting)
    ├── contrast.py         # WCAG contrast checker over the theme palettes
    ├── commands.py         # SlashCommandRegistry: menu metadata + dispatch in one place
    ├── screens/
    │   └── main.py         # Single-column chat screen + event handling + slash commands
    │   └── session_picker.py # Modal session switcher (/sessions)
    └── widgets/
        ├── chat.py         # Chat flow: user/assistant(Markdown)/tool cards/errors + scroll-follow
        ├── input.py        # Multi-line user input
        ├── status_bar.py   # Top status bar: spinner state + elapsed + tokens + model/workdir
        ├── tool_card.py    # Inline tool-call card (one-line summary, expandable)
        └── confirm.py      # Confirmation modal (ConfirmDialog, Confirmed/Rejected events)

scripts/
├── check_contrast.py       # Palette contrast report/CI gate (P2-3)
└── ui_walkthrough.py       # Render 5 key UI states to SVG for visual review (§7.4)

tests/
├── test_models.py
├── test_config.py
├── test_agent.py
├── test_llm_client.py
├── tools/
│   ├── test_read.py, test_bash.py, test_edit.py, test_write.py
│   ├── test_grep.py, test_find.py, test_ls.py, test_registry.py
└── ui/
    ├── test_app_smoke.py
    ├── test_confirmation.py
    └── test_widgets.py
```

## Data Flow

1. **User input** → `InputWidget` emits `UserSubmitted` → `MainScreen._handle_turn()`
2. **Agent** (`agent.py`) appends user message, calls `_conversation_loop()`
3. **LLM call** → `OpenAICompatibleClient.chat()` streams `TextChunk` / `ToolCallEvent`
4. **Text deltas** → rendered live in `ChatWidget` (streaming effect)
5. **Tool calls** → `ToolRegistry.execute(name, args, dry_run=True)` run dry first
6. **Confirmation-gated tools** (`edit`, `write`, `bash`): `ToolResult.requires_confirmation=True` pauses the loop; a `ConfirmDialog` modal is pushed
7. **User approves** → `Agent.apply_tool()` re-executes with `dry_run=False`
8. **User rejects** → `Agent.reject_pending_tool()` replaces placeholder with rejection message
9. **Loop continues** until LLM produces final text or `max_iterations` is hit
10. **Session saved** as JSONL (meta line + messages) to `~/.limbo/sessions/`
11. **Trace appended** throughout the run to `~/.limbo/sessions/traces/<id>.trace.jsonl` — full LLM request bodies, token usage (incl. cache hits), tool timing, confirmation decisions, errors. `/export` merges meta + trace + message snapshot into one JSONL

## AgentLoop Details

- `Agent.run()` yields `AgentEvent` types: `TextDelta`, `ThinkingDelta`, `ToolCallRequest`, `ToolResultEvent`, `ErrorEvent`, `UsageUpdate` (cumulative session token count, emitted after each LLM call)
- The loop respects `config.llm.max_iterations` (default 50)
- Multi-tool calls in a single assistant turn are executed sequentially
- When a tool requires confirmation, placeholder `role="tool"` messages are inserted for all remaining calls in that turn so the message history stays valid for the OpenAI API
- After a confirmed tool is applied, `Agent.continue_after_confirmation()` resumes the loop, first executing remaining placeholders, then returning to the LLM

## Tools

All tools extend `BaseTool` and implement `execute(arguments, dry_run=False) -> ToolResult`.

| Tool | Confirmation | Description |
|------|-------------|-------------|
| `read` | Never | Read file with `offset`/`limit`. Rejects paths outside workdir. Blocks sensitive files (`.env`, SSH keys, configurable). |
| `bash` | Always | Execute command with timeout. Safety filter blocks dangerous patterns (configurable). Output capped at 512KB. |
| `edit` | Always | Find exact `old_text` and replace with `new_text`. Requires uniqueness. Shows unified diff on confirmation. |
| `write` | Always | Create/overwrite file. Creates parent dirs automatically. |
| `grep` | Never | Search using ripgrep (preferred) or Python regex fallback. Respects `.gitignore`. Context lines require ripgrep. |
| `find` | Never | Glob-based file search. Respects root `.gitignore`. Built-in matcher supports basic gitignore rules. |
| `ls` | Never | List directory contents with dotfiles. Sorted: dirs first, then files. |

### Path Safety

All file tools use `resolve_path()` from `base.py`, which:
- Resolves the path via `Path.resolve()`
- Checks it's inside the workdir via `is_within_workdir()`
- Rejects broken symlinks when `strict=True` (default for read/edit)
- Returns `ToolResult(error=...)` on failure, never raises

## Configuration (`config.py`)

`~/.limbo/config.toml` is loaded by `load_config()`. Key sections:

```toml
[llm]
api_key = "..."           # Required
model = "deepseek-chat"   # Default
base_url = "https://api.deepseek.com/v1"
temperature = 0.2
max_iterations = 50

[tools]
bash_enabled = true

[ui]
theme = "limbo-dark"     # optional: limbo-dark (default) / limbo-light / any Textual built-in
show_banner = true       # startup ASCII art on fresh sessions

[safety]
dangerous_commands = ["rm", "git reset --hard"]
sensitive_files = [".env", "id_rsa", "id_ed25519", ".ssh"]

[providers.glm]      # optional per-provider override (base_url / api_key /
base_url = "..."     # api_key_env / headers); beats the global [llm] values
```

## UI Layout

Pi-style single-column layout (top to bottom):
- **Status bar (1 line)**: Agent state on the left (`● idle`, or animated `⠋ thinking… / running <tool>…` with elapsed seconds), model + cumulative tokens + workdir on the right
- **Chat flow**: The conversation as a single scrolling stream — user messages (`❯` prefix), assistant replies rendered as streaming Markdown, inline tool-call cards (`✓/⏸/✗` one-line summaries, click or `ctrl+o` to expand full output), error lines
- **Input box**: The only persistent rounded border on screen; Enter submits, Shift+Enter inserts a newline
- **Hint line (1 line)**: Key hints in muted color

Confirmation modal: `ConfirmDialog` shows tool output with `y`/`n`/`Esc` shortcuts and "Apply"/"Reject" buttons.

All styles live in `ui/app.tcss`; widgets do not define `DEFAULT_CSS`. Colors must
reference theme semantic variables only (no bare hex; 2048 tiles and the mascot
banner art are exempt). The palette and per-state color rules are defined in
`ui/theme.py` (RFC LIM-16): `limbo-dark` is the default, `limbo-light` is also
built in, and `[ui] theme` can select any Textual built-in theme — custom CSS
variables fall back to the limbo-dark values via `App.get_theme_variable_defaults()`.
Run `python scripts/check_contrast.py` after touching the palette: all *used*
text/background pairs must stay ≥ 4.5:1.

## Key Design Decisions

- **No provider field**: Provider is inferred from `base_url`, `model`, `api_key`
- **Async Agent, sync tools**: Tools run via `asyncio.to_thread()` — they are synchronous by default but executed in a thread pool
- **Session persistence**: Full conversation history rewritten on each save (safe for MVP-scale conversations). Sessions are resumable — see [docs/session-management.md](./docs/session-management.md): `limbo --continue` / `--resume <id>`, and in-TUI `/sessions`, `/new`, `/export` commands
- **System message**: Built into `Agent._init_system_message()` — tool descriptions and guidelines are hardcoded
- **Bash safety filter**: Heuristic only — tokenizes the command, checks against pattern list. Known bypass vectors (subshells, command substitution, variable indirection) are documented.

## Testing

```bash
pytest tests/ -v                     # all tests
ruff check src tests                 # lint
mypy src                             # type check (strict=false)
python scripts/check_contrast.py     # palette WCAG contrast gate
python scripts/ui_walkthrough.py     # export key UI states as SVGs to docs/assets/walkthrough/
```

Tests use `pytest-asyncio` for async tests, `respx` for HTTP mocking (LLM client),
temp directories for tool tests, and `pytest-textual-snapshot` for UI snapshots
(`pytest --snapshot-update` after intentional visual changes; review the SVG diff).

## Common Patterns

### Adding a new tool

1. Create `src/limbo/tools/<name>.py` with a class extending `BaseTool`
2. Set `name`, `description`, `parameters` (JSON schema)
3. Implement `run()` returning `ToolResult`; use `self.resolve_existing()` / `self.resolve_creatable()` for paths (they raise `ToolError`, which `execute()` converts to an error result)
4. Register in `ToolRegistry.__init__()` (add to the tool-class list or add conditional logic)
5. Add tests in `tests/tools/test_<name>.py`
6. If the tool needs config, wire it through `Config` and pass to the tool constructor in the registry
7. The tool will automatically appear in the LLM's tool definitions

### Confirmation flow for a new tool

Return `self.confirm(message)` from `run()` when `dry_run=True`. The agent loop and UI handle the rest.
