//! The agent loop: user input → LLM → tool calls → LLM → … → final answer.
//!
//! Each turn:
//! 1. loop top: compact if the next prompt would overflow, inject any
//!    steering messages the user queued, checkpoint the session file;
//! 2. stream one LLM response (retrying transient failures);
//! 3. no tool calls → the turn is over (unless a queued message keeps it
//!    going); otherwise journal each call's *intent*, run the batch
//!    concurrently, record results in source order, checkpoint, repeat.
//!
//! Frontends never touch the agent while a turn runs: the TUI *moves* the
//! `Agent` into a tokio task and gets it back when the turn ends. Events
//! flow out through a channel; steering and Esc flow in through a shared
//! queue and a cancellation token.
//!
//! 🦀 Rust notes
//! - Moving the agent into the task (instead of sharing it behind a lock)
//!   is the ownership system doing concurrency design for us: while a turn
//!   runs, nothing else *can* mutate the history.
//! - `FuturesUnordered` runs many futures concurrently and yields each
//!   result as soon as it finishes — perfect for parallel tool calls.

use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::attachments::build_user_content;
use crate::compaction::{
    self, build_summary_prompt, find_split_point, is_summary_message, make_summary_message,
    summary_body,
};
use crate::config::Config;
use crate::history;
use crate::llm::usage::{estimate_tokens, normalize_usage};
use crate::llm::{LlmClient, LlmError};
use crate::memory::{MEMORY_NUDGE, MemorySnapshot, MemoryStore, memory_dir};
use crate::models::{Attachment, LlmEvent, Message, Role, ToolCall, ToolResult, new_id};
use crate::prompt::{PromptExtras, build_system_prompt_with};
use crate::session::{self, CompactionRecord, LoadedSession, QueuedMessageRecord, SessionMeta};
use crate::subagents::Supervisor;
use crate::tools::{ToolContext, ToolRegistry};
use crate::trace::{TRACE_SCHEMA, TraceLogger, trace_path_for, utc_now_iso};

const MAX_CONSECUTIVE_LENGTH_STOPS: u32 = 3;

/// What the agent reports while it works.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TurnStart,
    Text(String),
    Thinking(String),
    /// The response streamed so far was discarded and is being re-sent.
    Retry {
        attempt: u32,
        reason: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    /// A line of live progress from a long-running tool (subagents).
    ToolProgress {
        id: String,
        line: String,
    },
    ToolResult {
        id: String,
        name: String,
        success: bool,
        output: String,
    },
    Compaction {
        compacted: bool,
        trigger: String,
        reason: Option<String>,
        tokens_before: u64,
        tokens_after: u64,
    },
    Usage {
        context_tokens: u64,
        context_window: u64,
        session_total: u64,
    },
    /// A queued message was injected into the conversation.
    Steer(String),
    /// A neutral notice (e.g. a background skill review started).
    Info(String),
    Warning(String),
    Error(String),
    Interrupted,
    TurnEnd,
    // -- task-level events (the pump) --
    GoalVerifyStarted {
        command: String,
        round: u32,
        max_rounds: u32,
    },
    GoalVerifyResult {
        exit_code: Option<i32>,
        output_tail: String,
        timed_out: bool,
        cancelled: bool,
        round: u32,
        max_rounds: u32,
    },
    GoalPassed {
        rounds: u32,
    },
    GoalExhausted {
        rounds: u32,
    },
    GoalResumed {
        max_rounds: u32,
    },
    /// The run failed and `[queue] after_failed_run` held (or withdrew) the queue.
    QueueHeld {
        policy: String,
        texts: Vec<String>,
    },
}

pub type EventTx = mpsc::UnboundedSender<AgentEvent>;

/// Messages the user typed while a turn was running.
///
/// Shared between the frontend (push) and the agent (drain) behind a
/// `Mutex`; every change is journaled so a crash never loses one.
#[derive(Default)]
pub struct SteerQueue {
    items: Mutex<VecDeque<QueuedMessageRecord>>,
    path: Mutex<Option<PathBuf>>,
}

pub const STEER: &str = "steer";
pub const FOLLOW_UP: &str = "follow_up";

impl SteerQueue {
    /// Queue a user message (steer: next consistency point; follow_up:
    /// after the whole task).
    pub fn push_item(
        &self,
        text: String,
        attachments: Vec<Attachment>,
        mode: &str,
        source: &str,
        refs: Vec<String>,
    ) -> String {
        let item = QueuedMessageRecord {
            id: new_id(),
            text,
            attachments,
            mode: mode.into(),
            source: source.into(),
            refs,
        };
        if let Some(p) = self.path.lock().unwrap().as_ref() {
            let _ = session::append_lines(p, &[session::queued_record(&item)]);
        }
        let id = item.id.clone();
        self.items.lock().unwrap().push_back(item);
        id
    }

    pub fn journal_dequeued(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        if let Some(p) = self.path.lock().unwrap().as_ref() {
            let _ = session::append_lines(p, &[session::dequeued_record(ids)]);
        }
    }

    /// Take the queued items of `modes` (FIFO). The caller journals them as
    /// dequeued once the message carrying them is on disk.
    pub fn drain_modes(&self, modes: &[&str]) -> Vec<QueuedMessageRecord> {
        let mut items = self.items.lock().unwrap();
        let (take, keep): (Vec<_>, Vec<_>) = items
            .drain(..)
            .partition(|i| modes.contains(&i.mode.as_str()));
        items.extend(keep);
        take
    }

    fn drain(&self) -> Vec<QueuedMessageRecord> {
        self.drain_modes(&[STEER])
    }

    /// Remove the most recently queued user message (the TUI puts it back
    /// into the input box).
    pub fn pop_last(&self) -> Option<String> {
        let item = {
            let mut items = self.items.lock().unwrap();
            let pos = items.iter().rposition(|i| i.source == "user")?;
            items.remove(pos)?
        };
        self.journal_dequeued(std::slice::from_ref(&item.id));
        Some(item.text)
    }

    /// Cancel one queued item by id (false once it was drained).
    pub fn cancel(&self, id: &str) -> Option<QueuedMessageRecord> {
        let item = {
            let mut items = self.items.lock().unwrap();
            let pos = items.iter().position(|i| i.id == id)?;
            items.remove(pos)?
        };
        self.journal_dequeued(std::slice::from_ref(&item.id));
        Some(item)
    }

    pub fn len(&self) -> usize {
        self.items.lock().unwrap().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn items(&self) -> Vec<QueuedMessageRecord> {
        self.items.lock().unwrap().iter().cloned().collect()
    }
}

pub struct Agent {
    pub config: Config,
    pub client: LlmClient,
    pub registry: ToolRegistry,
    pub workdir: PathBuf,
    pub session_dir: PathBuf,
    pub meta: SessionMeta,
    pub messages: Vec<Message>,
    compactions: Vec<CompactionRecord>,
    session_path: PathBuf,
    persisted: HashSet<String>,
    file_created: bool,
    needs_rewrite: bool,
    /// The skills catalog the current system prompt was built with.
    skills_catalog: String,
    pub steer: Arc<SteerQueue>,
    rerun: Vec<ToolCall>,
    /// Last real prompt size, and the history length it was measured at.
    last_prompt_tokens: Option<u64>,
    prompt_watermark: usize,
    last_finish_reason: Option<String>,
    /// Subagents send progress here when this agent is itself a child.
    pub is_subagent: bool,
    /// The memory this session's prompt was built with (None: memory off).
    pub memory: Option<MemorySnapshot>,
    turns_since_memory: u32,
    pub trace: TraceLogger,
    turn_count: u32,
    iteration: u32,
    /// The last turn ended because a tool asked for it (`ask_user`).
    pub ended_by_tool: bool,
    /// The last turn ended with an error (after retries).
    pub last_run_failed: bool,
    /// The last turn was interrupted by the user.
    pub was_interrupted: bool,
    /// Extra system-prompt blocks (project facts, subagent instructions).
    pub prompt_blocks: Vec<String>,
    /// The task board of this (main) session.
    pub supervisor: Option<Arc<Supervisor>>,
    /// Cache-parity fork children: the tool list sent to the model (the
    /// registry and its hooks still decide what runs).
    pub advertised_tools: Option<Vec<Value>>,
    /// Cache-parity fork children: the main session's system prompt verbatim.
    pub system_override: Option<String>,
    /// Daily mode: when to start a skill review by itself.
    pub learn: Arc<Mutex<crate::learn::LearnTrigger>>,
}

impl Agent {
    /// A fresh coding session.
    #[cfg(test)]
    pub fn new(config: Config, workdir: &Path, session_dir: &Path) -> anyhow::Result<Self> {
        Self::create(config, workdir, session_dir, false)
    }

    /// A fresh session, born in daily mode when `daily` is set (`limbo
    /// daily`): the very first system prompt, tool list and meta line are
    /// already the daily ones, as in the Python version.
    pub fn create(
        config: Config,
        workdir: &Path,
        session_dir: &Path,
        daily: bool,
    ) -> anyhow::Result<Self> {
        let client = LlmClient::new(&config)?;
        let id = new_session_id();
        let mut meta = SessionMeta::new(id.clone(), workdir, client.model_id());
        meta.daily_mode = daily;
        let path = session_dir.join(format!("{id}.jsonl"));
        let mut agent = Self::assemble(config, client, workdir, session_dir, meta, path);
        if daily {
            let cfg = agent.config.clone();
            agent.registry.set_daily(true, &cfg);
        }
        agent.attach_supervisor();
        agent.init_system_message();
        agent.log_session_start(false);
        Ok(agent)
    }

    /// Resume a session file, repairing tool calls a crash left dangling.
    pub fn resume(mut config: Config, path: &Path, session_dir: &Path) -> anyhow::Result<Self> {
        let LoadedSession {
            meta,
            messages,
            compactions,
            intents,
            queue,
            raw_message_ids,
        } = session::load_session_full(path)?;
        let workdir = PathBuf::from(&meta.workdir);
        let workdir = if workdir.is_dir() {
            workdir
        } else {
            std::env::current_dir()?
        };
        if !meta.model.is_empty() && config.llm.model == crate::config::LlmConfig::default().model {
            config.llm.model = meta.model.clone();
        }
        let client = LlmClient::new(&config)?;
        let mut agent = Self::assemble(
            config,
            client,
            &workdir,
            session_dir,
            meta.clone(),
            path.to_path_buf(),
        );
        agent.registry.plan_mode = meta.plan_mode;
        if meta.daily_mode {
            agent.registry.set_daily(true, &agent.config);
        }
        agent.registry.code_mode = meta.code_mode;
        agent.file_created = true;
        agent.persisted = raw_message_ids.into_iter().collect();
        agent.compactions = compactions;
        let started: HashSet<String> = intents.iter().map(|i| i.call_id.clone()).collect();
        let registry = agent.registry.clone();
        let rec = history::recover(messages, meta.tool_intents.then_some(&started), |name| {
            registry.get(name).is_some_and(|t| t.replay_safe())
        });
        agent.messages = rec.messages;
        agent.rerun = rec.rerun;
        for q in queue {
            agent.steer.items.lock().unwrap().push_back(q);
        }
        if agent.meta.role != "subagent" {
            agent.attach_supervisor();
        } else {
            agent.is_subagent = true;
        }
        agent.init_system_message();
        if rec.repaired {
            agent.needs_rewrite = true;
        }
        agent.log_session_start(true);
        Ok(agent)
    }

    /// A subagent: its own session file next to the parent's.
    pub fn new_subagent(
        config: Config,
        client: LlmClient,
        registry: ToolRegistry,
        workdir: &Path,
        session_path: PathBuf,
        parent: &str,
        task_id: &str,
    ) -> Self {
        let id = session_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(new_session_id);
        let mut meta = SessionMeta::new(id, workdir, client.model_id());
        meta.role = "subagent".into();
        meta.parent_session = Some(parent.to_string());
        meta.task_id = Some(task_id.to_string());
        let dir = session_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default();
        let mut agent = Self::assemble(config, client, workdir, &dir, meta, session_path);
        agent.registry = registry;
        agent.is_subagent = true;
        agent.init_system_message();
        agent
    }

    fn assemble(
        config: Config,
        mut client: LlmClient,
        workdir: &Path,
        session_dir: &Path,
        meta: SessionMeta,
        path: PathBuf,
    ) -> Self {
        let mut registry = ToolRegistry::for_config(&config);
        registry.session_dir = session_dir.to_path_buf();
        client.prompt_cache_key = Some(meta.id.clone());
        let memory = config.daily.memory.then(|| {
            MemoryStore::new(
                memory_dir(),
                config.daily.user_chars,
                config.daily.notes_chars,
            )
            .snapshot()
        });
        let trace = if cfg!(test) {
            TraceLogger::disabled()
        } else {
            TraceLogger::open(trace_path_for(&path))
        };
        let steer = Arc::new(SteerQueue::default());
        *steer.path.lock().unwrap() = Some(path.clone());
        let learn = Arc::new(Mutex::new(crate::learn::LearnTrigger::new(
            config.daily.learn_after_rounds,
            config.daily.learn_cooldown_turns,
        )));
        Self {
            config,
            client,
            registry,
            workdir: workdir.to_path_buf(),
            session_dir: session_dir.to_path_buf(),
            meta,
            messages: Vec::new(),
            compactions: Vec::new(),
            session_path: path,
            persisted: HashSet::new(),
            file_created: false,
            needs_rewrite: false,
            skills_catalog: String::new(),
            steer,
            rerun: Vec::new(),
            last_prompt_tokens: None,
            prompt_watermark: 0,
            last_finish_reason: None,
            is_subagent: false,
            memory,
            turns_since_memory: 0,
            trace,
            turn_count: 0,
            iteration: 0,
            ended_by_tool: false,
            last_run_failed: false,
            was_interrupted: false,
            prompt_blocks: Vec::new(),
            supervisor: None,
            advertised_tools: None,
            system_override: None,
            learn,
        }
    }

    /// Open the task board of the current session (main sessions only).
    fn attach_supervisor(&mut self) {
        if let Some(old) = self.supervisor.take() {
            old.stop();
        }
        let blocks_before = self.prompt_blocks.len();
        self.prompt_blocks
            .retain(|b| !b.starts_with(crate::subagents::supervisor::MAIN_PROMPT));
        for name in [
            "plan_tasks",
            "task_status",
            "task_send",
            "answer",
            "task_control",
        ] {
            self.registry.remove(name);
        }
        if !self.config.subagents.enabled || self.is_subagent {
            return;
        }
        let sup = Supervisor::new(
            self.config.clone(),
            &self.workdir,
            &self.session_path,
            self.client.model_id(),
            self.steer.clone(),
        );
        let tools = sup.main_tools();
        if !tools.is_empty() {
            for t in tools {
                self.registry.add(t);
            }
            self.prompt_blocks.push(sup.main_prompt());
        }
        let _ = blocks_before;
        self.supervisor = Some(sup);
    }

    /// Give the board what a fork child inherits.
    pub fn sync_supervisor(&self) {
        if let Some(sup) = &self.supervisor {
            sup.sync_main(
                &self.messages,
                self.registry.definitions(),
                self.daily_mode(),
            );
        }
    }

    fn resolved_llm_fields(&self) -> Value {
        json!({
            "model": self.client.spec.id,
            "provider": self.client.spec.provider.id,
            "api": format!("{:?}", self.client.spec.provider.dialect),
            "base_url": crate::llm::catalog::resolve_base_url(&self.client.spec, &self.config),
            "context_window": self.client.spec.context_window,
            "max_tokens": self.client.max_tokens,
            "thinking_effort": self.client.thinking_effort,
        })
    }

    fn log_session_start(&self, resumed: bool) {
        self.trace.log(
            "session_start",
            json!({
                "trace_schema": TRACE_SCHEMA,
                "session_id": self.meta.id,
                "workdir": self.workdir.display().to_string(),
                "resumed": resumed,
                "restored_messages": if resumed { self.messages.len().saturating_sub(1) } else { 0 },
                "limbo_version": env!("CARGO_PKG_VERSION"),
                "runtime": "rust",
                "platform": std::env::consts::OS,
                "config": {
                    "model": self.config.llm.model,
                    "base_url": self.config.llm.base_url,
                    "temperature": self.config.llm.temperature,
                    "max_tokens": self.config.llm.max_tokens,
                    "max_iterations": self.config.llm.max_iterations,
                    "thinking_effort": self.config.llm.thinking_effort,
                },
                "resolved": self.resolved_llm_fields(),
                "role": self.meta.role,
            }),
        );
    }

    // -- accessors ------------------------------------------------------------

    pub fn session_id(&self) -> &str {
        &self.meta.id
    }
    pub fn session_path(&self) -> &Path {
        &self.session_path
    }
    pub fn context_window(&self) -> u64 {
        self.client.spec.context_window
    }
    pub fn context_tokens(&self) -> u64 {
        self.estimated_next_prompt_tokens()
    }
    pub fn session_total_tokens(&self) -> u64 {
        self.meta.usage.values().map(|u| u.total_tokens).sum()
    }
    pub fn plan_mode(&self) -> bool {
        self.registry.plan_mode
    }
    pub fn daily_mode(&self) -> bool {
        self.registry.daily_mode
    }
    pub fn code_mode(&self) -> bool {
        self.registry.code_mode
    }
    /// The memory store `/memory` shows (None when `[daily] memory = false`).
    pub fn memory_store(&self) -> Option<MemoryStore> {
        let d = &self.config.daily;
        d.memory
            .then(|| MemoryStore::new(memory_dir(), d.user_chars, d.notes_chars))
    }

    fn current_skills_catalog(&self) -> String {
        crate::skills::format_skills(&crate::skills::discover_skills(
            &self.workdir,
            self.daily_mode(),
        ))
    }

    /// Rebuild the system prompt when skills were added, removed or changed
    /// since it was built, so a skill installed mid-session is visible to the
    /// model on the next turn. A no-op (prompt cache intact) otherwise.
    pub fn sync_skills(&mut self) {
        if self.system_override.is_some() {
            return;
        }
        if self.current_skills_catalog() != self.skills_catalog {
            self.init_system_message();
            self.needs_rewrite = true;
        }
    }

    /// (Re)build the system prompt as message 0.
    pub fn init_system_message(&mut self) {
        self.skills_catalog = self.current_skills_catalog();
        let extras = PromptExtras {
            memory: self.memory.as_ref(),
            blocks: self.prompt_blocks.clone(),
        };
        let text = match &self.system_override {
            Some(t) => t.clone(),
            None => build_system_prompt_with(&self.registry, &self.workdir, &extras),
        };
        let sys = Message::system(text);
        match self.messages.first_mut() {
            Some(m) if m.role == Role::System => {
                m.content = sys.content;
            }
            _ => self.messages.insert(0, sys),
        }
    }

    pub fn set_plan_mode(&mut self, on: bool) -> bool {
        if self.registry.plan_mode == on {
            return false;
        }
        self.registry.plan_mode = on;
        self.meta.plan_mode = on;
        self.mode_changed("plan_mode", on);
        true
    }

    /// Daily mode: swap the daily tools and the assistant prompt.
    pub fn set_daily_mode(&mut self, on: bool) -> bool {
        if self.registry.daily_mode == on {
            return false;
        }
        let cfg = self.config.clone();
        self.registry.set_daily(on, &cfg);
        self.meta.daily_mode = on;
        self.mode_changed("daily_mode", on);
        true
    }

    /// Code Mode: collapse the catalog to `run_code` (+ the SDK prompt).
    pub fn set_code_mode(&mut self, on: bool) -> bool {
        if self.registry.code_mode == on {
            return false;
        }
        self.registry.code_mode = on;
        self.meta.code_mode = on;
        self.mode_changed("code_mode", on);
        true
    }

    fn mode_changed(&mut self, kind: &str, on: bool) {
        self.init_system_message();
        self.needs_rewrite = true; // system prompt changed in place
        self.persist_meta();
        self.trace
            .log(kind, json!({"turn": self.turn_count, "enabled": on}));
    }

    /// Daily mode: a memory reminder every N quiet turns.
    fn memory_nudge(&mut self) -> &'static str {
        use std::sync::atomic::Ordering;
        if self.registry.memory_written.swap(false, Ordering::SeqCst) {
            self.turns_since_memory = 0;
        }
        self.turns_since_memory += 1;
        let every = self.config.daily.memory_nudge_turns;
        if !(self.daily_mode() && every > 0 && self.registry.get("memory").is_some()) {
            return "";
        }
        if self.turns_since_memory <= every {
            return "";
        }
        self.turns_since_memory = 0;
        MEMORY_NUDGE
    }

    /// Persist the goal state at once (a crash mid-loop keeps progress).
    pub fn save_goal(&mut self, goal: Option<crate::goal::GoalState>) {
        self.meta.goal = goal;
        self.persist_meta();
        if !self.file_created {
            // nothing on disk yet: the first checkpoint writes the meta
        }
    }

    /// Swap the model mid-session (the history stays).
    pub fn set_model(&mut self, model: &str) -> anyhow::Result<()> {
        let mut cfg = self.config.clone();
        cfg.llm.model = model.to_string();
        let mut client = LlmClient::new(&cfg)?;
        client.prompt_cache_key = Some(self.meta.id.clone());
        self.client = client;
        self.config = cfg;
        self.meta.model = model.to_string();
        self.last_prompt_tokens = None;
        self.persist_meta();
        self.trace.log(
            "model_switch",
            json!({"turn": self.turn_count, "resolved": self.resolved_llm_fields()}),
        );
        Ok(())
    }

    pub fn set_thinking(&mut self, level: Option<String>) {
        self.config.llm.thinking_effort = level.clone();
        self.client.thinking_effort = level;
    }

    pub fn persist_meta(&mut self) {
        if self.file_created {
            let _ = session::append_lines(
                &self.session_path,
                &[session::meta_patch_record(&self.meta)],
            );
        }
    }

    // -- persistence ------------------------------------------------------------

    /// Append everything not yet on disk (or rewrite the file when history
    /// changed in place). Compaction summaries are not appended: loading
    /// rebuilds them from the `compaction` record.
    pub fn checkpoint(&mut self) {
        if self.meta.title.is_empty() {
            let t = session::derive_title(&self.messages);
            if !t.is_empty() {
                self.meta.title = t;
                if self.file_created {
                    let _ = session::rewrite_meta_line(&self.session_path, &mut self.meta);
                }
            }
        }
        if !self.messages.iter().any(|m| m.role == Role::User) {
            return; // nothing worth a file yet
        }
        if !self.file_created || self.needs_rewrite {
            match session::save_session(
                &self.session_path,
                &mut self.meta,
                &self.messages,
                &self.compactions,
            ) {
                Ok(()) => {
                    self.file_created = true;
                    self.needs_rewrite = false;
                    self.persisted = self.messages.iter().map(|m| m.id.clone()).collect();
                }
                Err(e) => eprintln!("warning: could not save session: {e}"),
            }
            return;
        }
        let fresh: Vec<Value> = self
            .messages
            .iter()
            .filter(|m| !self.persisted.contains(&m.id) && !is_summary_message(m))
            .map(session::message_line)
            .collect();
        if fresh.is_empty() {
            return;
        }
        if session::append_lines(&self.session_path, &fresh).is_err() {
            self.needs_rewrite = true;
            return self.checkpoint();
        }
        for m in &self.messages {
            self.persisted.insert(m.id.clone());
        }
    }

    fn journal(&self, records: &[Value]) {
        if self.file_created {
            let _ = session::append_lines(&self.session_path, records);
        }
    }

    // -- token accounting ------------------------------------------------------

    fn estimated_next_prompt_tokens(&self) -> u64 {
        match self.last_prompt_tokens {
            Some(p) => {
                p + estimate_tokens(
                    &self.messages[self.prompt_watermark.min(self.messages.len())..],
                )
            }
            None => estimate_tokens(&self.messages),
        }
    }

    fn record_usage(&mut self, usage: Option<&Value>) {
        let totals = normalize_usage(usage);
        let key = format!("{}/{}", self.client.spec.provider.id, self.client.spec.id);
        let entry = self.meta.usage.entry(key).or_default();
        entry.requests += 1;
        entry.prompt_tokens += totals.prompt_tokens.unwrap_or(0);
        entry.cached_tokens += totals.cached_tokens.unwrap_or(0);
        entry.total_tokens += totals.total_tokens.unwrap_or(0);
        if let Some(p) = totals.prompt_tokens {
            self.last_prompt_tokens = Some(p);
            self.prompt_watermark = self.messages.len();
        }
    }

    fn usage_event(&self) -> AgentEvent {
        AgentEvent::Usage {
            context_tokens: self.estimated_next_prompt_tokens(),
            context_window: self.context_window(),
            session_total: self.session_total_tokens(),
        }
    }

    // -- the turn ------------------------------------------------------------

    /// Run one user turn to completion (or interruption).
    pub async fn run(
        &mut self,
        input: String,
        attachments: Vec<Attachment>,
        tx: &EventTx,
        cancel: &CancellationToken,
    ) {
        let _ = tx.send(AgentEvent::TurnStart);
        self.turn_count += 1;
        self.iteration = 0;
        self.ended_by_tool = false;
        self.last_run_failed = false;
        self.was_interrupted = false;
        let turn_started_at = utc_now_iso();
        let turn_start = std::time::Instant::now();
        // Read-only calls interrupted by a crash are re-run first.
        if !self.rerun.is_empty() {
            let calls = std::mem::take(&mut self.rerun);
            self.trace.log(
                "tool_recovery",
                json!({"turn": self.turn_count, "rerun_tool_call_ids": calls.iter().map(|c| c.id.clone()).collect::<Vec<_>>()}),
            );
            for tc in calls {
                let ctx = self.tool_ctx(cancel, &tc.id, tx);
                let result = self
                    .registry
                    .execute(tc.name(), tc.function.arguments.clone(), &ctx)
                    .await;
                if let Some(m) = self
                    .messages
                    .iter_mut()
                    .find(|m| m.tool_call_id.as_deref() == Some(&tc.id))
                {
                    m.content = Some(result.content_for_model());
                    m.attachments = result.attachments.clone();
                }
                self.needs_rewrite = true;
            }
        }
        // Messages queued before this turn started run first.
        for item in self.steer.drain() {
            self.inject_steer(item, tx);
        }
        if !input.trim().is_empty() || !attachments.is_empty() {
            let (mut content, images) =
                build_user_content(&input, &attachments, self.client.spec.vision);
            content.push_str(self.memory_nudge());
            self.trace.log(
                "user_message",
                json!({"turn": self.turn_count, "content": content}),
            );
            let mut m = Message::user(content);
            if !images.is_empty() {
                m.attachments = Some(images);
            }
            self.messages.push(m);
        }
        self.conversation_loop(tx, cancel).await;
        let status = if self.was_interrupted || cancel.is_cancelled() {
            "interrupted"
        } else {
            "completed"
        };
        self.trace.log(
            "turn_end",
            json!({
                "turn": self.turn_count,
                "started_at": turn_started_at,
                "duration": turn_start.elapsed().as_secs_f64(),
                "iterations": self.iteration,
                "status": status,
            }),
        );
        self.checkpoint();
        let _ = tx.send(self.usage_event());
        let _ = tx.send(AgentEvent::TurnEnd);
    }

    fn inject_steer(&mut self, item: QueuedMessageRecord, tx: &EventTx) {
        if item.source == "user" {
            let _ = tx.send(AgentEvent::Steer(item.text.clone()));
        }
        self.trace.log(
            "user_message",
            json!({"turn": self.turn_count, "steer": true, "followup": false, "id": item.id, "content": item.text}),
        );
        let (content, images) =
            build_user_content(&item.text, &item.attachments, self.client.spec.vision);
        let mut m = Message::user(content);
        if !images.is_empty() {
            m.attachments = Some(images);
        }
        self.messages.push(m);
        // Mark it dequeued only once the message carrying it is on disk.
        self.checkpoint();
        self.journal(&[session::dequeued_record(&[item.id])]);
    }

    fn tool_ctx(&self, cancel: &CancellationToken, call_id: &str, tx: &EventTx) -> ToolContext {
        ToolContext {
            workdir: self.workdir.clone(),
            cancel: cancel.clone(),
            call_id: call_id.to_string(),
            events: Some(tx.clone()),
            registry: Some(self.registry.clone()),
        }
    }

    async fn conversation_loop(&mut self, tx: &EventTx, cancel: &CancellationToken) {
        let max_iterations = self.config.llm.max_iterations;
        let mut iteration = 0;
        let mut length_stops = 0;
        let mut auto_compaction_failed = false;
        let mut overflow_recovered = false;
        while iteration < max_iterations {
            iteration += 1;
            self.iteration += 1;
            if !auto_compaction_failed
                && compaction::should_compact(
                    self.estimated_next_prompt_tokens(),
                    self.context_window(),
                    self.config.compaction.enabled,
                    self.config.compaction.reserve_tokens,
                )
                && !self.compact("auto", tx, cancel).await
            {
                auto_compaction_failed = true;
            }
            if cancel.is_cancelled() {
                break;
            }
            for item in self.steer.drain() {
                self.inject_steer(item, tx);
            }
            self.checkpoint();

            match self.call_llm(tx, cancel).await {
                Ok(true) => {}
                Ok(false) => {
                    // interrupted mid-stream
                    self.mark_interrupted("llm_stream");
                    let _ = tx.send(AgentEvent::Interrupted);
                    break;
                }
                Err(e) => {
                    self.trace.log(
                        "llm_error",
                        json!({"turn": self.turn_count, "iteration": self.iteration, "error": e.to_string(), "context_overflow": e.is_context_overflow()}),
                    );
                    if e.is_context_overflow() && !overflow_recovered {
                        overflow_recovered = true;
                        if self.compact("overflow", tx, cancel).await && !cancel.is_cancelled() {
                            iteration -= 1;
                            continue;
                        }
                    }
                    self.last_run_failed = true;
                    let _ = tx.send(AgentEvent::Error(e.friendly()));
                    return;
                }
            }
            self.checkpoint();
            let _ = tx.send(self.usage_event());

            let last = self.messages.last().cloned().expect("assistant message");
            let truncated = matches!(
                self.last_finish_reason.as_deref(),
                Some("length" | "max_tokens")
            );
            if !last.has_tool_calls() {
                if truncated {
                    let _ = tx.send(AgentEvent::Warning(format!(
                        "Model stopped because it reached the output token limit (max_tokens={}); the response above may be incomplete.",
                        self.client.max_tokens
                    )));
                }
                if cancel.is_cancelled() {
                    break;
                }
                // Messages that arrived while the answer streamed keep the
                // turn alive as fresh input.
                let pending = self.steer.drain();
                if pending.is_empty() {
                    break;
                }
                iteration = 0;
                length_stops = 0;
                for item in pending {
                    self.inject_steer(item, tx);
                }
                continue;
            }
            let calls = last.tool_calls.clone().unwrap_or_default();

            if truncated {
                // Tool arguments may be cut off: fail the whole batch unrun.
                length_stops += 1;
                for tc in &calls {
                    let msg = format!(
                        "Tool call not executed: the response hit the output token limit (max_tokens={}) so its \
                         arguments may be truncated. Write large content to a file in smaller pieces with write, \
                         or split the task into smaller steps, instead of re-issuing the same call.",
                        self.client.max_tokens
                    );
                    let _ = tx.send(AgentEvent::ToolResult {
                        id: tc.id.clone(),
                        name: tc.name().into(),
                        success: false,
                        output: msg.clone(),
                    });
                    self.messages.push(Message::tool(tc.id.clone(), msg));
                }
                if length_stops >= MAX_CONSECUTIVE_LENGTH_STOPS {
                    let _ = tx.send(AgentEvent::Error(format!(
                        "Stopped after {length_stops} consecutive responses that hit the output token limit \
                         (max_tokens={}): the model keeps re-issuing a tool call that does not fit in one response. \
                         Ask it to write the script to a file with the write tool and run it via bash, split the \
                         task into smaller steps, or switch to a model with a higher output limit.",
                        self.client.max_tokens
                    )));
                    break;
                }
                continue;
            }

            if iteration >= max_iterations {
                for tc in &calls {
                    self.messages.push(Message::tool(
                        tc.id.clone(),
                        "Maximum iteration count reached; tool not executed.",
                    ));
                }
                let _ = tx.send(AgentEvent::Error(
                    "Maximum iteration count reached; remaining tool calls were cancelled.".into(),
                ));
                break;
            }
            length_stops = 0;
            if calls
                .iter()
                .any(|c| matches!(c.name(), "plan_tasks" | "task_control"))
            {
                self.sync_supervisor();
            }
            self.execute_tool_calls(&calls, tx, cancel).await;
            self.checkpoint();
            if cancel.is_cancelled() {
                self.mark_interrupted("tools");
                let _ = tx.send(AgentEvent::Interrupted);
                break;
            }
            if self.ended_by_tool {
                self.trace
                    .log("turn_ended_by_tool", json!({"turn": self.turn_count}));
                break;
            }
        }
    }

    fn mark_interrupted(&mut self, phase: &str) {
        self.was_interrupted = true;
        self.trace.log(
            "turn_interrupted",
            json!({"turn": self.turn_count, "iteration": self.iteration, "phase": phase}),
        );
    }

    /// Stream one response into a new assistant message.
    /// `Ok(false)` = interrupted by the user.
    async fn call_llm(
        &mut self,
        tx: &EventTx,
        cancel: &CancellationToken,
    ) -> Result<bool, LlmError> {
        let tools = self
            .advertised_tools
            .clone()
            .unwrap_or_else(|| self.registry.definitions());
        let max_retries = self.config.llm.max_retries;
        let mut attempt = 0;
        loop {
            let mut client = self.client.clone();
            if self.trace.enabled() {
                let trace = self.trace.clone();
                let (turn, iteration, att) = (self.turn_count, self.iteration, attempt + 1);
                client.on_request = Some(Arc::new(move |body: &Value| {
                    trace.log(
                        "llm_request",
                        json!({"turn": turn, "iteration": iteration, "attempt": att, "body": body}),
                    );
                }));
            }
            let started_at = utc_now_iso();
            let mut rx = client.chat(self.messages.clone(), tools.clone());
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut signature: Option<String> = None;
            let mut calls: Vec<ToolCall> = Vec::new();
            let mut meta = None;
            let mut failure: Option<LlmError> = None;
            loop {
                let item = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => None,
                    item = rx.recv() => Some(item),
                };
                let Some(item) = item else {
                    // Keep what was said so far; the partial answer stays in history.
                    if !text.is_empty() || !reasoning.is_empty() {
                        let mut m = Message::assistant(format!("{text}\n\n[interrupted by user]"));
                        if !reasoning.is_empty() {
                            m.reasoning = Some(reasoning);
                        }
                        self.messages.push(m);
                    }
                    return Ok(false);
                };
                match item {
                    None => break,
                    Some(Ok(LlmEvent::Text(t))) => {
                        text.push_str(&t);
                        let _ = tx.send(AgentEvent::Text(t));
                    }
                    Some(Ok(LlmEvent::Thinking {
                        text: t,
                        signature: s,
                    })) => {
                        if !t.is_empty() {
                            reasoning.push_str(&t);
                            let _ = tx.send(AgentEvent::Thinking(t));
                        }
                        if let Some(s) = s {
                            signature = Some(match signature.take() {
                                // Anthropic streams signatures in pieces.
                                Some(prev)
                                    if !s.starts_with(crate::llm::responses::SIGNATURE_PREFIX) =>
                                {
                                    prev + &s
                                }
                                _ => s,
                            });
                        }
                    }
                    Some(Ok(LlmEvent::ToolCall {
                        id,
                        name,
                        arguments,
                    })) => calls.push(ToolCall::new(id, name, arguments)),
                    Some(Ok(LlmEvent::Done(m))) => meta = Some(m),
                    Some(Err(e)) => {
                        failure = Some(e);
                        break;
                    }
                }
            }
            if let Some(e) = failure {
                if e.is_retryable() && attempt < max_retries {
                    attempt += 1;
                    self.trace.log(
                        "llm_stream_retry",
                        json!({"turn": self.turn_count, "iteration": self.iteration, "attempt": attempt, "error": e.to_string(), "discarded_chars": text.chars().count(), "discarded_tool_calls": calls.len()}),
                    );
                    let _ = tx.send(AgentEvent::Retry {
                        attempt,
                        reason: e.to_string(),
                    });
                    let delay = self.config.llm.retry_base_delay * 2f64.powi(attempt as i32 - 1);
                    tokio::select! {
                        _ = tokio::time::sleep(std::time::Duration::from_secs_f64(delay.min(30.0))) => {}
                        _ = cancel.cancelled() => return Ok(false),
                    }
                    continue;
                }
                return Err(e);
            }
            let meta = meta.unwrap_or_default();
            let totals = normalize_usage(meta.usage.as_ref());
            self.trace.log(
                "llm_response",
                json!({
                    "turn": self.turn_count,
                    "iteration": self.iteration,
                    "started_at": started_at,
                    "duration": meta.duration,
                    "ttft": meta.ttft,
                    "finish_reason": meta.finish_reason,
                    "usage": meta.usage,
                    "cached_tokens": totals.cached_tokens,
                    "content_chars": text.chars().count(),
                    "reasoning_chars": reasoning.chars().count(),
                    "content": text,
                    "reasoning": reasoning,
                    "tool_calls": calls.iter().map(|c| json!({"id": c.id, "name": c.name(), "arguments": c.function.arguments})).collect::<Vec<_>>(),
                }),
            );
            let mut m = Message::assistant(text);
            if !reasoning.is_empty() {
                m.reasoning = Some(reasoning);
            }
            m.reasoning_signature = signature;
            if !calls.is_empty() {
                m.tool_calls = Some(calls);
            }
            self.messages.push(m);
            self.last_finish_reason = meta.finish_reason.clone();
            self.record_usage(meta.usage.as_ref());
            self.persist_meta();
            return Ok(true);
        }
    }

    /// Run one batch of tool calls (concurrently unless `[tools] parallel =
    /// false`). Results stream in completion order but are recorded in the
    /// order the model asked for them.
    async fn execute_tool_calls(
        &mut self,
        calls: &[ToolCall],
        tx: &EventTx,
        cancel: &CancellationToken,
    ) {
        // Commit each call's intent before any side effect runs.
        let intents: Vec<Value> = calls
            .iter()
            .map(|tc| {
                let tool = self.registry.get(tc.name());
                session::tool_intent_record(&tc.id, tc.name(), tool.as_deref())
            })
            .collect();
        self.journal(&intents);
        for tc in calls {
            let _ = tx.send(AgentEvent::ToolCall {
                id: tc.id.clone(),
                name: tc.name().into(),
                arguments: tc.function.arguments.clone(),
            });
            self.trace.log(
                "tool_call",
                json!({"turn": self.turn_count, "iteration": self.iteration, "id": tc.id, "name": tc.name(), "arguments": tc.function.arguments}),
            );
        }
        let started_at = utc_now_iso();
        let batch_start = std::time::Instant::now();
        let mut results: Vec<Option<ToolResult>> = vec![None; calls.len()];
        if self.config.tools.parallel {
            let mut running = FuturesUnordered::new();
            for (i, tc) in calls.iter().enumerate() {
                let registry = self.registry.clone();
                let ctx = self.tool_ctx(cancel, &tc.id, tx);
                let tc = tc.clone();
                running.push(async move {
                    let r = registry
                        .execute(tc.name(), tc.function.arguments.clone(), &ctx)
                        .await;
                    (i, r)
                });
            }
            while let Some((i, r)) = running.next().await {
                let tc = &calls[i];
                let _ = tx.send(AgentEvent::ToolResult {
                    id: tc.id.clone(),
                    name: tc.name().into(),
                    success: r.success,
                    output: r.content_for_model(),
                });
                results[i] = Some(r);
            }
        } else {
            for (i, tc) in calls.iter().enumerate() {
                let r = if cancel.is_cancelled() {
                    ToolResult::err("Interrupted by the user before this call ran.")
                } else {
                    let ctx = self.tool_ctx(cancel, &tc.id, tx);
                    self.registry
                        .execute(tc.name(), tc.function.arguments.clone(), &ctx)
                        .await
                };
                let _ = tx.send(AgentEvent::ToolResult {
                    id: tc.id.clone(),
                    name: tc.name().into(),
                    success: r.success,
                    output: r.content_for_model(),
                });
                results[i] = Some(r);
            }
        }
        let vision = self.client.spec.vision;
        for (tc, r) in calls.iter().zip(results) {
            let r = r.unwrap_or_else(|| ToolResult::err("tool did not run"));
            self.trace.log(
                "tool_result",
                json!({
                    "turn": self.turn_count, "iteration": self.iteration, "id": tc.id, "name": tc.name(),
                    "success": r.success, "output": r.output, "error": r.error,
                    "started_at": started_at, "duration": batch_start.elapsed().as_secs_f64(),
                }),
            );
            if r.end_turn {
                self.ended_by_tool = true;
            }
            let mut m = Message::tool(tc.id.clone(), r.content_for_model());
            if vision {
                m.attachments = r.attachments;
            } else if let Some(atts) = r.attachments {
                // No vision: degrade images to a path reference.
                let paths: Vec<String> = atts.iter().map(|a| a.path.clone()).collect();
                m.content = Some(format!(
                    "{}\n[image not shown: current model has no vision; file: {}]",
                    m.text(),
                    paths.join(", ")
                ));
            }
            self.messages.push(m);
        }
    }

    // -- compaction ------------------------------------------------------------

    /// Summarize older history. Returns whether anything was compacted.
    pub async fn compact(
        &mut self,
        trigger: &str,
        tx: &EventTx,
        cancel: &CancellationToken,
    ) -> bool {
        let before = self.estimated_next_prompt_tokens();
        let Some(split) =
            find_split_point(&self.messages, self.config.compaction.keep_recent_tokens)
        else {
            let _ = tx.send(AgentEvent::Compaction {
                compacted: false,
                trigger: trigger.into(),
                reason: Some("没有可压缩的历史".into()),
                tokens_before: before,
                tokens_after: before,
            });
            return false;
        };
        let head_len = usize::from(
            self.messages
                .first()
                .is_some_and(|m| m.role == Role::System),
        );
        let region = &self.messages[head_len..split];
        let previous = region.iter().find_map(summary_body);
        let to_summarize: Vec<Message> = region
            .iter()
            .filter(|m| !is_summary_message(m))
            .cloned()
            .collect();
        let prompt = build_summary_prompt(&to_summarize, previous.as_deref());
        let summary = tokio::select! {
            s = self.generate_summary(prompt) => s,
            _ = cancel.cancelled() => Err(LlmError::Other("interrupted".into())),
        };
        let summary = match summary {
            Ok(s) if !s.trim().is_empty() => s,
            Ok(_) | Err(_) => {
                let reason = match summary {
                    Err(e) => format!("压缩失败：{e}"),
                    _ => "压缩失败：模型返回了空摘要".into(),
                };
                let _ = tx.send(AgentEvent::Compaction {
                    compacted: false,
                    trigger: trigger.into(),
                    reason: Some(reason),
                    tokens_before: before,
                    tokens_after: before,
                });
                return false;
            }
        };
        let first_kept = self.messages[split].id.clone();
        let mut new_messages: Vec<Message> = self.messages[..head_len].to_vec();
        new_messages.push(make_summary_message(&summary));
        new_messages.extend_from_slice(&self.messages[split..]);
        self.messages = new_messages;
        let after = estimate_tokens(&self.messages);
        let record = CompactionRecord {
            id: new_id(),
            created_at: session::now_iso(),
            first_kept_entry_id: first_kept,
            summary,
            tokens_before: Some(before),
            tokens_after_estimate: Some(after),
            trigger: trigger.into(),
        };
        self.checkpoint();
        self.journal(&[session::compaction_record(&record)]);
        self.trace.log(
            "compaction",
            json!({"turn": self.turn_count, "trigger": trigger, "tokens_before": before, "tokens_after": after, "first_kept_entry_id": record.first_kept_entry_id}),
        );
        self.compactions.push(record);
        self.last_prompt_tokens = None;
        self.prompt_watermark = 0;
        let _ = tx.send(AgentEvent::Compaction {
            compacted: true,
            trigger: trigger.into(),
            reason: None,
            tokens_before: before,
            tokens_after: after,
        });
        let _ = tx.send(self.usage_event());
        true
    }

    async fn generate_summary(&mut self, prompt: Vec<Message>) -> Result<String, LlmError> {
        let mut rx = self.client.chat(prompt, vec![]);
        let mut text = String::new();
        while let Some(item) = rx.recv().await {
            match item? {
                LlmEvent::Text(t) => text.push_str(&t),
                LlmEvent::Done(m) => self.record_usage_ledger_only(m.usage.as_ref()),
                _ => {}
            }
        }
        Ok(text)
    }

    fn record_usage_ledger_only(&mut self, usage: Option<&Value>) {
        let totals = normalize_usage(usage);
        let key = format!("{}/{}", self.client.spec.provider.id, self.client.spec.id);
        let e = self.meta.usage.entry(key).or_default();
        e.requests += 1;
        e.prompt_tokens += totals.prompt_tokens.unwrap_or(0);
        e.cached_tokens += totals.cached_tokens.unwrap_or(0);
        e.total_tokens += totals.total_tokens.unwrap_or(0);
    }

    // -- lineage ------------------------------------------------------------

    fn user_turn_indices(&self) -> Vec<usize> {
        self.messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::User && !is_summary_message(m))
            .map(|(i, _)| i)
            .collect()
    }

    fn child_meta(&self, title: String) -> SessionMeta {
        let mut meta = SessionMeta::new(new_session_id(), &self.workdir, &self.meta.model);
        meta.title = title;
        meta.parent_session = Some(self.meta.id.clone());
        meta.plan_mode = self.meta.plan_mode;
        meta.daily_mode = self.meta.daily_mode;
        meta.code_mode = self.meta.code_mode;
        meta
    }

    /// Copy the history (the first `keep_turns` user turns, or all) into a
    /// new, self-contained session file. The parent is untouched.
    pub fn fork(&mut self, keep_turns: Option<usize>) -> anyhow::Result<PathBuf> {
        self.checkpoint();
        let end = match keep_turns {
            Some(n) => {
                let idx = self.user_turn_indices();
                if n == 0 || n > idx.len() {
                    anyhow::bail!("fork: turn {n} out of range (1-{})", idx.len());
                }
                idx.get(n).copied().unwrap_or(self.messages.len())
            }
            None => self.messages.len(),
        };
        let mut copied: Vec<Message> = self.messages[..end].to_vec();
        // Never end a fork on an assistant tool call without results.
        let rec = history::recover(copied.split_off(0), None, |_| false);
        let mut msgs = vec![self.messages[0].clone()];
        msgs.extend(rec.messages);
        let mut meta = self.child_meta(format!("⑂ {}", self.meta.title));
        meta.fork_at = msgs.last().map(|m| m.id.clone());
        let path = self.session_dir.join(format!("{}.jsonl", meta.id));
        session::save_session(&path, &mut meta, &msgs, &[])?;
        Ok(path)
    }

    /// Start a clean, linked session from a note (the brief is the first
    /// user message of the new session).
    pub fn handoff_prompt(&self, note: &str) -> String {
        format!(
            "This session continues work from an earlier session ({}, \"{}\"). The handoff note from that session:\n\n{note}",
            self.meta.id, self.meta.title
        )
    }

    pub fn handoff_meta(&self) -> SessionMeta {
        self.child_meta(String::new())
    }

    /// Replace this agent's session with a brand-new one (same config/model).
    pub fn reset_to(&mut self, meta: SessionMeta) {
        let path = self.session_dir.join(format!("{}.jsonl", meta.id));
        self.client.prompt_cache_key = Some(meta.id.clone());
        let cfg = self.config.clone();
        self.registry.plan_mode = meta.plan_mode;
        self.registry.set_daily(meta.daily_mode, &cfg);
        self.registry.code_mode = meta.code_mode;
        self.meta = meta;
        if !cfg!(test) {
            self.trace = TraceLogger::open(trace_path_for(&path));
        }
        self.session_path = path.clone();
        *self.steer.path.lock().unwrap() = Some(path);
        self.messages.clear();
        self.compactions.clear();
        self.persisted.clear();
        self.file_created = false;
        self.needs_rewrite = false;
        self.rerun.clear();
        self.last_prompt_tokens = None;
        self.prompt_watermark = 0;
        self.attach_supervisor();
        self.init_system_message();
    }

    pub fn new_session(&mut self) {
        let mut meta = SessionMeta::new(new_session_id(), &self.workdir, &self.meta.model);
        // The posture survives /new; plan mode and the goal do not.
        meta.daily_mode = self.meta.daily_mode;
        meta.code_mode = self.meta.code_mode;
        self.reset_to(meta);
        self.log_session_start(false);
    }
}

/// `20261102-021814-ab12cd` — sortable and readable.
pub fn new_session_id() -> String {
    let short: String = new_id().chars().take(6).collect();
    format!("{}-{short}", chrono::Local::now().format("%Y%m%d-%H%M%S"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A minimal fake OpenAI-compatible server: each request gets the next
    /// canned SSE body. Returns the base URL and the captured request bodies.
    pub async fn fake_server(bodies: Vec<String>) -> (String, Arc<Mutex<Vec<Value>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        tokio::spawn(async move {
            let mut bodies = bodies.into_iter();
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                // Read headers + body (Content-Length).
                loop {
                    let n = sock.read(&mut chunk).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some(h) = text.find("\r\n\r\n") {
                        let len = text[..h]
                            .lines()
                            .find_map(|l| {
                                l.to_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        if buf.len() >= h + 4 + len {
                            let body: Value = serde_json::from_slice(&buf[h + 4..h + 4 + len])
                                .unwrap_or(Value::Null);
                            seen2.lock().unwrap().push(body);
                            break;
                        }
                    }
                }
                let body = bodies.next().unwrap_or_default();
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        (format!("http://{addr}/v1"), seen)
    }

    pub fn sse(chunks: &[Value]) -> String {
        let mut s: String = chunks.iter().map(|c| format!("data: {c}\n\n")).collect();
        s.push_str("data: [DONE]\n\n");
        s
    }

    pub fn test_config(base: &str) -> Config {
        let mut c = Config::default();
        c.llm.model = "test-model".into();
        c.llm.base_url = base.into();
        c.llm.api_key = Some("k".into());
        c.llm.max_retries = 0;
        c
    }

    #[test]
    fn web_search_notice_only_in_daily_mode_when_enabled_but_missing() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        let mut cfg = test_config("http://127.0.0.1:9/v1");
        // An unknown provider never registers, whatever keys this machine has.
        cfg.web_search.provider = "nope".into();
        let mut agent = Agent::new(cfg, dir.path(), &sessions).unwrap();
        assert!(crate::commands::web_search_missing_notice(&agent).is_none());
        agent.set_daily_mode(true);
        let n = crate::commands::web_search_missing_notice(&agent).unwrap();
        assert!(n.contains("\"nope\""));
        agent.config.web_search.enabled = false;
        assert!(crate::commands::web_search_missing_notice(&agent).is_none());
    }

    #[tokio::test]
    async fn full_turn_with_tool_call_and_resume() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "hi there").unwrap();
        let first = sse(&[
            serde_json::json!({"choices":[{"delta":{"content":"Let me look."}}]}),
            serde_json::json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"path\":\"hello.txt\"}"}}]},"finish_reason":"tool_calls"}]}),
            serde_json::json!({"choices":[],"usage":{"prompt_tokens":100,"total_tokens":120}}),
        ]);
        let second = sse(&[
            serde_json::json!({"choices":[{"delta":{"content":"It says hi."},"finish_reason":"stop"}]}),
            serde_json::json!({"choices":[],"usage":{"prompt_tokens":150,"total_tokens":160}}),
        ]);
        let (base, seen) = fake_server(vec![first, second]).await;
        let sessions = dir.path().join("sessions");
        let mut agent = Agent::new(test_config(&base), dir.path(), &sessions).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        agent
            .run(
                "what is in hello.txt?".into(),
                vec![],
                &tx,
                &CancellationToken::new(),
            )
            .await;
        drop(tx);
        let mut events = vec![];
        while let Some(e) = rx.recv().await {
            events.push(e);
        }
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolResult { success: true, output, .. } if output == "hi there")));
        assert!(matches!(events.last(), Some(AgentEvent::TurnEnd)));
        // system, user, assistant(tool call), tool, assistant
        assert_eq!(agent.messages.len(), 5);
        assert_eq!(agent.messages[4].text(), "It says hi.");
        assert_eq!(agent.session_total_tokens(), 280);
        // The second request carried the tool result.
        let bodies = seen.lock().unwrap().clone();
        assert_eq!(bodies[1]["messages"][3]["role"], "tool");
        assert_eq!(
            bodies[1]["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"hello.txt\"}"
        );

        // Resume from disk: same history, title derived.
        let path = agent.session_path().to_path_buf();
        let resumed = Agent::resume(test_config(&base), &path, &sessions).unwrap();
        assert_eq!(resumed.messages.len(), 5);
        assert_eq!(resumed.meta.title, "what is in hello.txt?");
        assert_eq!(resumed.meta.usage["custom/test-model"].requests, 2);
    }

    #[tokio::test]
    async fn crash_mid_batch_is_repaired_on_resume() {
        let dir = tempfile::tempdir().unwrap();
        let sessions = dir.path().join("sessions");
        let mut agent =
            Agent::new(test_config("http://127.0.0.1:9/v1"), dir.path(), &sessions).unwrap();
        agent.messages.push(Message::user("go"));
        let mut a = Message::assistant("");
        a.tool_calls = Some(vec![ToolCall::new(
            "w1",
            "write",
            serde_json::json!({"path": "x", "content": "y"}),
        )]);
        agent.messages.push(a);
        agent.checkpoint();
        agent.journal(&[session::tool_intent_record(
            "w1",
            "write",
            Some(&crate::tools::write::WriteTool),
        )]);
        // "crash" here, then resume
        let resumed = Agent::resume(
            test_config("http://127.0.0.1:9/v1"),
            agent.session_path(),
            &sessions,
        )
        .unwrap();
        let last = resumed.messages.last().unwrap();
        assert_eq!(last.role, Role::Tool);
        assert_eq!(last.text(), history::PARTIAL_CONTENT);
    }

    #[test]
    fn daily_session_is_born_daily_and_resumes_in_its_own_mode() {
        let dir = tempfile::tempdir().unwrap();
        // Sessions outside the workdir: the prompt's project facts count files.
        let store = tempfile::tempdir().unwrap();
        let sessions = store.path().join("sessions");
        let mut cfg = test_config("http://127.0.0.1:9/v1");
        cfg.daily.memory = false;
        let has_fetch = |a: &Agent| a.registry.get("web_fetch").is_some();

        // `limbo daily`: daily from the first system prompt and meta line.
        let mut daily = Agent::create(cfg.clone(), dir.path(), &sessions, true).unwrap();
        assert!(daily.daily_mode());
        assert!(daily.meta.daily_mode);
        assert!(has_fetch(&daily));
        assert!(!daily.needs_rewrite);
        let daily_prompt = daily.messages[0].text().to_string();
        daily.messages.push(Message::user("hi"));
        daily.checkpoint();

        // A plain session stays a coding one.
        let mut code = Agent::new(cfg.clone(), dir.path(), &sessions).unwrap();
        assert!(!code.daily_mode());
        assert_ne!(code.messages[0].text(), daily_prompt);
        code.messages.push(Message::user("hi"));
        code.checkpoint();

        // Resuming restores each session's own mode.
        let d = Agent::resume(cfg.clone(), daily.session_path(), &sessions).unwrap();
        assert!(d.daily_mode());
        assert!(has_fetch(&d));
        assert_eq!(d.messages[0].text(), daily_prompt);
        let c = Agent::resume(cfg, code.session_path(), &sessions).unwrap();
        assert!(!c.daily_mode());
        assert!(!has_fetch(&c));
    }

    #[tokio::test]
    async fn manual_compaction_appends_record() {
        let dir = tempfile::tempdir().unwrap();
        let summary = sse(&[
            serde_json::json!({"choices":[{"delta":{"content":"## Goal\nship it"},"finish_reason":"stop"}]}),
        ]);
        let (base, _) = fake_server(vec![summary]).await;
        let sessions = dir.path().join("sessions");
        let mut cfg = test_config(&base);
        cfg.compaction.keep_recent_tokens = 1000;
        let mut agent = Agent::new(cfg.clone(), dir.path(), &sessions).unwrap();
        for i in 0..6 {
            agent
                .messages
                .push(Message::user(format!("question {i} {}", "x".repeat(2000))));
            agent
                .messages
                .push(Message::assistant(format!("answer {i}")));
        }
        agent.checkpoint();
        let (tx, _rx) = mpsc::unbounded_channel();
        assert!(
            agent
                .compact("manual", &tx, &CancellationToken::new())
                .await
        );
        assert!(is_summary_message(&agent.messages[1]));
        let n = agent.messages.len();
        let resumed = Agent::resume(cfg, agent.session_path(), &sessions).unwrap();
        assert_eq!(resumed.messages.len(), n);
        assert!(resumed.messages[1].text().contains("ship it"));
    }
}
