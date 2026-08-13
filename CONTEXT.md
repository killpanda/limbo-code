# Context — Limbo domain glossary

Domain language for Limbo's concepts. Architecture reviews and refactor
proposals should name modules with these terms.

## Token accounting

- **Usage** — the provider-reported token counters on one completion. Each
  dialect reports them differently (OpenAI `prompt_tokens`, Anthropic
  `input_tokens` + cache counters, DeepSeek `prompt_cache_hit_tokens`,
  Responses `input_tokens_details`). Normalized by `llm/usage.py` into
  `UsageTotals` (prompt / cached / total); the agent loop and compaction
  never read provider field names.
- **Prompt-size estimation** — predicting the *next* request's prompt size:
  the last real usage figure plus the chars/4 estimate of everything
  appended since (the **watermark**). Owned by `PromptSizeEstimator` in
  `llm/usage.py`; invalidated (reset) when compaction rewrites history.

## Conversation

- **Turn** — one user input through the agent loop until a tool-call-free
  response (or max_iterations). Turns are the save/trace unit.
- **Turn pump** — the non-UI module that runs turns back-to-back
  (`pump.py`, `TurnPump`): pulls events out of the lazy `Agent.run()`
  stream, owns the single-flight busy flag (turns and `/compact`
  mini-turns alike) and the goal closed loop. The follow-up steer drain
  lives in its supervisor (stragglers past a round's boundary drain —
  verify/wrap-up window, teardown race — become one more turn); frontends
  never drain. Frontends are adapters translating its events into UI
  updates.
- **Steer** — a user message queued mid-turn, injected at consistency
  points (loop top, turn end, run head). The `SteerQueue` (`steer.py`) owns
  queueing semantics — id generation, FIFO, and the cancel boundary (an
  item can be cancelled by id only until drained); the **Turn pump** owns
  drain timing.
- **Compaction** — summarizing the old region of history into a synthetic
  summary message, keeping the recent tail. Triggered automatically
  (loop-top, before context overflow) or manually (`/compact`).
- **Attachment policy** — how a submitted attachment reaches the model
  (`attachments.py`): images become multimodal blocks for vision models
  (encoded per dialect, `llm/scaffold.py`), everything else degrades to
  text — small files inline (≤50KB), the rest as path references.
- **Model switch** — `/model` mid-session (`model_switch.py`): validate the
  target (API key, thinking-effort compatibility), swap the client
  (converging on the latest config model), persist to config.toml. The UI
  screen is an adapter translating verdicts into chat messages.

## Safety

Limbo has no safety fences: file tools are not scoped to the workdir,
there is no sensitive-file list, and bash runs every command as given.
Only run it with trusted models in repositories you can afford to modify
or lose.
