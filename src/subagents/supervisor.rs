//! The supervisor: runs the task board of one main session.
//!
//! - **plans** become tasks on a board persisted next to the session
//!   (`<session>.tasks.json`), with dependencies, priorities and path
//!   claims; under `[subagents] auto = "suggest"` the model's plan waits for
//!   the user's approval;
//! - **reconcile** after every change: dependencies that finished make a
//!   task ready, failed ones notify (or cascade), ready tasks start while
//!   fewer than `max_parallel` run;
//! - **runners**: each running task is a child [`Agent`] with its own
//!   session file under `<main id>.agents/`, driven by a tokio task;
//! - **notices** go to the main agent through its steer queue
//!   (`source = "subagent"`) as one `<subagent_updates>` message, waking
//!   an idle main agent for questions, alerts and finished waves.
//!
//! 🦀 The supervisor is shared (`Arc`) between the main agent's tools, the
//! runners and the frontends; its mutable state sits behind `Mutex`es that
//! are never held across an `.await`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::policies::{AllowListPolicy, ClaimPolicy, NoEditPolicy, ReadOnlyPolicy};
use super::profiles::{
    NO_EDIT, Profile, ProjectFacts, READ_ONLY, discover_profiles, format_facts,
    format_profiles_for_prompt, resolve_profile, scan_project,
};
use super::{tools as task_tools, worktree};
use crate::agent::{Agent, AgentEvent, FOLLOW_UP, STEER, SteerQueue};
use crate::background::{self, BackgroundEvent};
use crate::config::Config;
use crate::llm::LlmClient;
use crate::models::{Message, Role};
use crate::session::{self, SessionMeta};
use crate::tools::ToolRegistry;

pub const TERMINAL: [&str; 3] = ["done", "failed", "cancelled"];

pub fn status_label(s: &str) -> &str {
    match s {
        "proposed" => "待确认",
        "pending" => "排队",
        "ready" => "就绪",
        "running" => "运行中",
        "waiting" => "等待",
        "paused" => "已暂停",
        "done" => "完成",
        "failed" => "失败",
        "cancelled" => "已取消",
        other => other,
    }
}

pub const MAIN_PROMPT: &str = "## Subagents

You can delegate work to background subagents with plan_tasks. They run in
parallel with their own context while you keep working with the user; their
results, questions and failures come back to you as <subagent_updates>
messages. Manage them with task_status, task_send, answer and task_control.

Delegate when:
- two or more independent parts of the code must change;
- a broad search or reading would flood your context and you only need the
  conclusion (use the explore profile);
- an independent review or a long verification can run beside the work.
Do not delegate small single-file changes, strictly sequential steps where
every step needs the previous result, work that needs most of this
conversation's context, or anything the user is actively discussing with you.

Give every task a self-contained brief (subagents cannot see this
conversation), declare scope_paths for tasks that write files, and express
ordering with depends_on. When the user asks to pause, resume, cancel, retry,
reprioritize or redirect a subagent, use task_control / task_send. Answer
subagent questions with answer; ask the user when the decision is theirs.";

/// Talking to the user and curating memory/skills stay with the main agent.
const MAIN_ONLY_TOOLS: [&str; 4] = ["memory", "ask_user", "skill_manage", "session_search"];

fn child_prompt(agent: &str, id: &str, title: &str, scope: &str, others: &str) -> String {
    format!(
        "<subagent_role>
You are the subagent \"{agent}\" of limbo, working on task {id}: {title}.
The main agent coordinates the work; you cannot see its conversation and the
user does not watch you live.
- Call checkpoint(phase, summary) when you move to a new phase.
- If your task turns out to depend on another task's result, call
  need(on=[task ids], reason). If it needs work nobody owns yet, call
  need(new={{title, brief}}, reason). Do not do other tasks' work, and do
  not edit paths another task claimed.
- If a decision belongs to the main agent or the user, call ask_main.
- When you are done, call finish(summary, files_changed); the summary is all
  the main agent reads. If you cannot complete the task, call
  finish(status=\"failed\") and say why.
{scope}{others}</subagent_role>"
    )
}

fn fork_child_prompt(agent: &str, id: &str, title: &str, tools: &str) -> String {
    format!(
        "<subagent_role>
From here on you are the subagent \"{agent}\" of limbo, working on task {id}:
{title}. The conversation above is the main session's, shown for context;
the user does not watch you live.
- The tool list is the main session's, but you may only call: {tools}.
  Any other tool fails.
- When you are done, reply with your result as plain text and no tool
  calls: that reply is all the main agent reads.
</subagent_role>"
    )
}

// -- the board ---------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    pub seq: u32,
    pub phase: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub progress: Option<f64>,
    #[serde(default)]
    pub auto: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Question {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub answer: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Outcome {
    pub status: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub files_changed: Vec<String>,
    #[serde(default)]
    pub follow_ups: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub brief: String,
    pub profile: String,
    pub status: String,
    pub depends_on: Vec<String>,
    pub created_by: String,
    pub acceptance: String,
    pub scope_paths: Vec<String>,
    pub priority: i64,
    pub isolation: String,
    pub context: String,
    pub on_dep_fail: String,
    pub max_tokens: u64,
    pub checkpoints: Vec<Checkpoint>,
    pub questions: Vec<Question>,
    pub inbox: Vec<String>,
    pub resume_note: String,
    pub waiting_on: Vec<String>,
    pub wait_reason: String,
    pub paused_from: String,
    pub outcome: Option<Outcome>,
    pub session_file: Option<String>,
    pub worktree: Option<String>,
    pub branch: Option<String>,
    pub tokens: u64,
    pub attempt: u32,
    pub dep_alerted: bool,
}

impl Task {
    pub fn terminal(&self) -> bool {
        TERMINAL.contains(&self.status.as_str())
    }
    pub fn agent_name(&self) -> &str {
        &self.profile
    }
    pub fn last_checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }
    pub fn open_question(&self) -> Option<&Question> {
        self.questions.iter().find(|q| q.answer.is_none())
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct BoardFile {
    tasks: Vec<Task>,
    next_id: u32,
    reported: Vec<String>,
}

#[derive(Default)]
struct Board {
    tasks: Vec<Task>,
    next_id: u32,
    reported: HashSet<String>,
}

impl Board {
    fn get(&self, id: &str) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }
    fn get_mut(&mut self, id: &str) -> Option<&mut Task> {
        self.tasks.iter_mut().find(|t| t.id == id)
    }
    /// A task by id, or the live task of a profile name (`@worker`).
    fn find(&self, r: &str) -> Option<&Task> {
        let r = r.trim().trim_start_matches('@');
        self.get(r).or_else(|| {
            let live: Vec<&Task> = self.tasks.iter().filter(|t| !t.terminal()).collect();
            live.iter()
                .rev()
                .find(|t| t.profile == r)
                .copied()
                .or_else(|| self.tasks.iter().rev().find(|t| t.profile == r))
        })
    }
    fn live(&self) -> Vec<&Task> {
        self.tasks.iter().filter(|t| !t.terminal()).collect()
    }
    fn with_status(&self, statuses: &[&str]) -> Vec<&Task> {
        self.tasks
            .iter()
            .filter(|t| statuses.contains(&t.status.as_str()))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Notice {
    pub id: String,
    pub kind: String,
    pub text: String,
    pub headline: String,
}

const DELIVERED: [&str; 3] = ["result", "alert", "question"];

pub struct PlanResult {
    pub message: String,
    pub proposed: bool,
    pub ids: Vec<String>,
}

struct Runner {
    cancel: CancellationToken,
    steer: Arc<SteerQueue>,
    messages: Arc<Mutex<Vec<Message>>>,
}

/// What a fork child inherits from the main agent.
#[derive(Default, Clone)]
struct MainSnapshot {
    messages: Vec<Message>,
    tools: Vec<Value>,
    daily: bool,
}

pub struct Supervisor {
    me: Weak<Supervisor>,
    pub config: Config,
    pub workdir: PathBuf,
    main_session: PathBuf,
    main_id: String,
    main_model: String,
    main_steer: Arc<SteerQueue>,
    pub profiles: BTreeMap<String, Profile>,
    facts: OnceLock<ProjectFacts>,
    board: Mutex<Board>,
    runners: Mutex<HashMap<String, Runner>>,
    pending: Mutex<Vec<Notice>>,
    activity: Mutex<HashMap<String, String>>,
    main_busy: AtomicBool,
    main: Mutex<MainSnapshot>,
    stopped: AtomicBool,
    deadlocks: Mutex<HashSet<String>>,
    pub interrupted_on_load: Vec<String>,
}

fn join_notes(a: &str, b: &str) -> String {
    [a, b]
        .iter()
        .filter(|s| !s.trim().is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("\n")
}

fn last_answer(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::Assistant && !m.text().trim().is_empty())
        .map(|m| m.text().trim().to_string())
        .unwrap_or_default()
}

impl Supervisor {
    /// The board of `main_session`, loaded from disk when it exists.
    pub fn new(
        config: Config,
        workdir: &Path,
        main_session: &Path,
        main_model: &str,
        main_steer: Arc<SteerQueue>,
    ) -> Arc<Self> {
        let main_id = main_session
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let file: BoardFile = std::fs::read_to_string(Self::board_path_for(main_session))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        let mut board = Board {
            tasks: file.tasks,
            next_id: file.next_id,
            reported: file.reported.into_iter().collect(),
        };
        // Notices restored in the main queue were delivered (dedupe).
        for item in main_steer.items() {
            if item.source == "subagent" {
                board.reported.extend(item.refs);
            }
        }
        // Tasks that were running when the process stopped.
        let mut interrupted = Vec::new();
        let note = "（进程重启，你的工作被中断）先检查工作区现状，再从上次打点继续。";
        for t in board.tasks.iter_mut().filter(|t| t.status == "running") {
            interrupted.push(t.id.clone());
            t.resume_note = join_notes(&t.resume_note, note);
            if config.subagents.on_resume == "auto" {
                t.status = "ready".into();
            } else {
                t.status = "paused".into();
                t.paused_from = "running".into();
            }
        }
        let sup = Arc::new_cyclic(|me| Self {
            me: me.clone(),
            profiles: discover_profiles(workdir),
            config,
            workdir: workdir.to_path_buf(),
            main_session: main_session.to_path_buf(),
            main_id,
            main_model: main_model.to_string(),
            main_steer,
            facts: OnceLock::new(),
            board: Mutex::new(board),
            runners: Mutex::new(HashMap::new()),
            pending: Mutex::new(Vec::new()),
            activity: Mutex::new(HashMap::new()),
            main_busy: AtomicBool::new(false),
            main: Mutex::new(MainSnapshot::default()),
            stopped: AtomicBool::new(false),
            deadlocks: Mutex::new(HashSet::new()),
            interrupted_on_load: interrupted.clone(),
        });
        if !interrupted.is_empty() && sup.config.subagents.on_resume != "auto" {
            sup.notice(
                "",
                "info",
                "",
                &format!(
                    "{} 个子任务在上次退出时中断（{}），已暂停；/task resume all 继续",
                    interrupted.len(),
                    interrupted.join("、")
                ),
                None,
            );
        }
        sup.regenerate_notices();
        sup
    }

    fn arc(&self) -> Arc<Self> {
        self.me.upgrade().expect("supervisor alive")
    }

    fn board_path_for(main_session: &Path) -> PathBuf {
        let stem = main_session
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        main_session.with_file_name(format!("{stem}.tasks.json"))
    }

    pub fn agents_dir(&self) -> PathBuf {
        self.main_session
            .with_file_name(format!("{}.agents", self.main_id))
    }

    pub fn facts(&self) -> &ProjectFacts {
        self.facts.get_or_init(|| scan_project(&self.workdir))
    }

    /// The main agent's system-prompt section.
    pub fn main_prompt(&self) -> String {
        [
            MAIN_PROMPT.to_string(),
            format_profiles_for_prompt(&self.profiles),
            format_facts(self.facts()),
        ]
        .join("\n\n")
    }

    /// Tools the main agent gets (none with `auto = "off"`).
    pub fn main_tools(&self) -> Vec<Arc<dyn crate::tools::Tool>> {
        if self.config.subagents.auto == "off" {
            return vec![];
        }
        task_tools::main_tools(self.arc())
    }

    fn save(&self, board: &Board) {
        let file = BoardFile {
            tasks: board.tasks.clone(),
            next_id: board.next_id,
            reported: board.reported.iter().cloned().collect(),
        };
        if board.tasks.is_empty() && !Self::board_path_for(&self.main_session).exists() {
            return;
        }
        if let Ok(text) = serde_json::to_string_pretty(&file) {
            let _ =
                session::atomic_write(&Self::board_path_for(&self.main_session), text.as_bytes());
        }
    }

    /// Mutate the board, persist, announce, reconcile.
    fn update<R>(&self, f: impl FnOnce(&mut Board) -> R) -> R {
        let (r, changed) = {
            let mut b = self.board.lock().unwrap();
            let before: Vec<(String, String)> = b
                .tasks
                .iter()
                .map(|t| (t.id.clone(), t.status.clone()))
                .collect();
            let r = f(&mut b);
            self.save(&b);
            let changed: Vec<(String, String, String)> = b
                .tasks
                .iter()
                .filter(|t| !before.iter().any(|(id, s)| *id == t.id && *s == t.status))
                .map(|t| (t.id.clone(), t.status.clone(), t.title.clone()))
                .collect();
            (r, changed)
        };
        for (id, status, title) in changed {
            background::emit_for(
                &self.main_id,
                BackgroundEvent::TaskChanged { id, status, title },
            );
        }
        r
    }

    // -- main agent state ---------------------------------------------------------

    pub fn set_main_busy(&self, busy: bool) {
        self.main_busy.store(busy, Ordering::SeqCst);
        if !busy {
            self.reconcile();
        }
    }

    pub fn main_busy(&self) -> bool {
        self.main_busy.load(Ordering::SeqCst)
    }

    /// What a fork child inherits (called by the main agent before tools run).
    pub fn sync_main(&self, messages: &[Message], tools: Vec<Value>, daily: bool) {
        *self.main.lock().unwrap() = MainSnapshot {
            messages: messages.to_vec(),
            tools,
            daily,
        };
    }

    // -- scheduling -------------------------------------------------------------

    pub fn reconcile(&self) {
        if self.stopped.load(Ordering::SeqCst) {
            return;
        }
        let max_parallel = self.config.subagents.max_parallel.max(1);
        let mut to_start = Vec::new();
        let mut alerts: Vec<(String, String, String)> = Vec::new();
        let mut cancels: Vec<(String, String)> = Vec::new();
        self.update(|b| {
            // 1. dependencies
            let snapshot: HashMap<String, (String, String)> = b
                .tasks
                .iter()
                .map(|t| {
                    let summary = t
                        .outcome
                        .as_ref()
                        .map(|o| o.summary.clone())
                        .unwrap_or_default();
                    (t.id.clone(), (t.status.clone(), summary))
                })
                .collect();
            for t in b.tasks.iter_mut() {
                let deps: &[String] = match t.status.as_str() {
                    "pending" => &t.depends_on,
                    "waiting" if !t.waiting_on.is_empty() => &t.waiting_on,
                    _ => continue,
                };
                let failed: Vec<&String> = deps
                    .iter()
                    .filter(|d| {
                        snapshot
                            .get(*d)
                            .is_none_or(|(s, _)| s == "failed" || s == "cancelled")
                    })
                    .collect();
                if !failed.is_empty() {
                    let names = failed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
                    if t.on_dep_fail == "cascade" {
                        cancels.push((t.id.clone(), format!("依赖 {names} 失败或被取消")));
                    } else if !t.dep_alerted {
                        t.dep_alerted = true;
                        alerts.push((
                            t.id.clone(),
                            format!(
                                "任务 {} 的依赖 {names} 失败或被取消，它不会开始。用 task_control 决定：retry 依赖、edit 去掉依赖，或 cancel 它。",
                                t.id
                            ),
                            format!("{} 的依赖 {names} 失败", t.id),
                        ));
                    }
                    continue;
                }
                if deps.iter().all(|d| snapshot.get(d).is_some_and(|(s, _)| s == "done")) {
                    let notes: Vec<String> = deps
                        .iter()
                        .filter_map(|d| {
                            let (_, summary) = snapshot.get(d)?;
                            let s: String = summary.chars().take(1500).collect();
                            (!s.is_empty()).then(|| format!("[{d} 的结果]\n{s}"))
                        })
                        .collect();
                    t.resume_note = join_notes(&t.resume_note, &notes.join("\n\n"));
                    t.status = "ready".into();
                    t.waiting_on.clear();
                    t.wait_reason.clear();
                }
            }
            // 2. start ready tasks by priority
            let running = b.tasks.iter().filter(|t| t.status == "running").count();
            let mut ready: Vec<&Task> = b.tasks.iter().filter(|t| t.status == "ready").collect();
            ready.sort_by_key(|t| -t.priority);
            for t in ready.into_iter().take(max_parallel.saturating_sub(running)) {
                to_start.push(t.id.clone());
            }
            for id in &to_start {
                if let Some(t) = b.get_mut(id) {
                    t.status = "running".into();
                }
            }
        });
        for (id, reason) in cancels {
            self.cancel_task(&id, &reason, "system");
        }
        for (id, text, headline) in alerts {
            self.notice(&id, "alert", &text, &headline, None);
        }
        self.detect_deadlock();
        for id in to_start {
            self.start(id);
        }
        self.deliver();
    }

    fn detect_deadlock(&self) {
        let chain: Vec<String> = {
            let b = self.board.lock().unwrap();
            if !b.with_status(&["running", "ready"]).is_empty() {
                return;
            }
            let blocked: Vec<&Task> = b
                .tasks
                .iter()
                .filter(|t| t.status == "waiting" && !t.waiting_on.is_empty())
                .collect();
            if blocked.is_empty()
                || !blocked.iter().all(|t| {
                    t.waiting_on.iter().all(|d| {
                        b.get(d)
                            .is_some_and(|x| x.status == "waiting" || x.status == "pending")
                    })
                })
            {
                return;
            }
            blocked
                .iter()
                .map(|t| format!("{} 等待 {}", t.id, t.waiting_on.join(", ")))
                .collect()
        };
        let sig = chain.join(";");
        if !self.deadlocks.lock().unwrap().insert(sig) {
            return;
        }
        self.notice(
            "",
            "alert",
            &format!(
                "子任务互相等待，没有任何任务能继续：\n{}\n请取消其中一个或用 task_control(edit) 调整依赖。",
                chain.join("\n")
            ),
            &format!("子任务死锁：{}", chain.join("；")),
            None,
        );
    }

    // -- runners -------------------------------------------------------------------

    fn model_for(&self, p: &Profile) -> String {
        match p.model.as_str() {
            "" | "inherit" => self.main_model.clone(),
            "cheap" => {
                if self.config.subagents.cheap_model.is_empty() {
                    self.main_model.clone()
                } else {
                    self.config.subagents.cheap_model.clone()
                }
            }
            m => m.to_string(),
        }
    }

    fn inherits_prompt(&self, t: &Task, p: &Profile) -> bool {
        t.context == "fork" && p.inherit_prompt && self.model_for(p) == self.main_model
    }

    fn start(&self, id: String) {
        let sup = self.arc();
        let cancel = CancellationToken::new();
        let steer = Arc::new(SteerQueue::default());
        let messages = Arc::new(Mutex::new(Vec::new()));
        self.runners.lock().unwrap().insert(
            id.clone(),
            Runner {
                cancel: cancel.clone(),
                steer: steer.clone(),
                messages: messages.clone(),
            },
        );
        tokio::spawn(async move {
            let result = sup.run(&id, cancel, steer, messages).await;
            if let Err(e) = result {
                let terminal = sup
                    .board
                    .lock()
                    .unwrap()
                    .get(&id)
                    .is_none_or(|t| t.terminal());
                if !terminal {
                    sup.finish(&id, "", vec![], vec![], true, &format!("子代理异常：{e}"));
                }
            }
            sup.runners.lock().unwrap().remove(&id);
            sup.activity.lock().unwrap().remove(&id);
            sup.reconcile();
        });
    }

    fn open_agent(&self, task: &Task, steer: Arc<SteerQueue>) -> anyhow::Result<(Agent, bool)> {
        let profile = resolve_profile(&self.profiles, &task.profile);
        let mut cfg = self.config.clone();
        cfg.llm.max_iterations = profile.max_iterations;
        cfg.llm.model = self.model_for(&profile);
        cfg.subagents.enabled = false; // depth 1
        let client = LlmClient::new(&cfg).or_else(|_| {
            cfg.llm.model = self.main_model.clone();
            LlmClient::new(&cfg)
        })?;
        let workdir = task
            .worktree
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.workdir.clone());
        let path = task
            .session_file
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                self.agents_dir()
                    .join(format!("{}-{}.jsonl", task.id, task.agent_name()))
            });
        let resumed = path.exists();
        let main = self.main.lock().unwrap().clone();
        if !resumed && task.context == "fork" {
            let mut meta = SessionMeta::new(path_stem(&path), &self.workdir, &cfg.llm.model);
            meta.title = format!("{} · {}", task.id, task.title);
            meta.role = "subagent".into();
            meta.task_id = Some(task.id.clone());
            meta.parent_session = Some(self.main_id.clone());
            meta.daily_mode = main.daily;
            // An idle main agent's history is complete; mid-turn, stop
            // before the half-finished turn (never a dangling tool batch).
            let mut kept = main.messages.clone();
            if self.main_busy()
                && let Some(cut) = kept
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(i, m)| *i > 0 && m.role == Role::User)
                    .map(|(i, _)| i + 1)
            {
                kept.truncate(cut);
            }
            meta.fork_at = kept.last().map(|m| m.id.clone());
            session::save_session(&path, &mut meta, &kept, &[])?;
        }
        let mut agent = if path.exists() {
            Agent::resume(cfg.clone(), &path, path.parent().unwrap_or(Path::new(".")))?
        } else {
            let mut a = Agent::new_subagent(
                cfg.clone(),
                client,
                ToolRegistry::for_config(&cfg),
                &workdir,
                path.clone(),
                &self.main_id,
                &task.id,
            );
            a.meta.title = format!("{} · {}", task.id, task.title);
            if main.daily {
                a.set_daily_mode(true);
            }
            a
        };
        agent.is_subagent = true;
        agent.steer = steer;
        if let Some(r) = self.runners.lock().unwrap().get(&task.id) {
            let _ = r; // the runner keeps the steer handle registered in start()
        }
        // -- tools & policies per profile
        let names = profile.tool_names();
        let remove: Vec<String> = match &names {
            Some(allowed) => agent
                .registry
                .all_names()
                .into_iter()
                .filter(|n| !allowed.iter().any(|a| a == n))
                .map(String::from)
                .collect(),
            None => MAIN_ONLY_TOOLS.iter().map(|s| s.to_string()).collect(),
        };
        if profile.tools == READ_ONLY {
            agent.registry.policies.push(Arc::new(ReadOnlyPolicy));
        } else if profile.tools == NO_EDIT {
            agent.registry.policies.push(Arc::new(NoEditPolicy));
        }
        if profile.writes() && task.isolation == "shared" {
            agent.registry.policies.push(Arc::new(ClaimPolicy {
                task_id: task.id.clone(),
                sup: self.me.clone(),
                workdir: workdir.clone(),
            }));
        }
        if self.inherits_prompt(task, &profile) {
            // Advertising is not permission: the main tool list is shown,
            // the policy decides what runs.
            agent.registry.policies.push(Arc::new(AllowListPolicy {
                allowed: names.unwrap_or_default(),
            }));
            agent.system_override = main
                .messages
                .first()
                .filter(|m| m.role == Role::System)
                .map(|m| m.text().to_string());
            agent.advertised_tools = Some(main.tools.clone());
            agent.client.prompt_cache_key = Some(self.main_id.clone());
        } else {
            for n in &remove {
                agent.registry.remove(n);
            }
            for t in task_tools::subagent_tools(self.arc(), &task.id) {
                agent.registry.add(t);
            }
            let b = self.board.lock().unwrap();
            let scope = if task.scope_paths.is_empty() {
                String::new()
            } else {
                format!(
                    "- Your task claims these paths: {}\n",
                    task.scope_paths.join(", ")
                )
            };
            let others: Vec<String> = b
                .live()
                .into_iter()
                .filter(|t| t.id != task.id)
                .take(20)
                .map(|t| {
                    let claims = if t.scope_paths.is_empty() {
                        String::new()
                    } else {
                        format!(" — claims {}", t.scope_paths.join(", "))
                    };
                    format!(
                        "  {} ({}, {}): {}{claims}",
                        t.id,
                        t.agent_name(),
                        t.status,
                        t.title
                    )
                })
                .collect();
            let others = if others.is_empty() {
                String::new()
            } else {
                format!("- Other tasks on the board:\n{}\n", others.join("\n"))
            };
            let mut block = child_prompt(task.agent_name(), &task.id, &task.title, &scope, &others);
            if !profile.body.is_empty() {
                block.push_str("\n\n");
                block.push_str(&profile.body);
            }
            agent.prompt_blocks = vec![block];
        }
        agent.init_system_message();
        Ok((agent, resumed))
    }

    fn first_input(&self, task: &Task, resumed: bool) -> String {
        let inbox = task
            .inbox
            .iter()
            .map(|m| format!("- {m}"))
            .collect::<Vec<_>>()
            .join("\n");
        if resumed {
            let mut parts = vec![format!(
                "（任务继续）{}",
                if task.resume_note.is_empty() {
                    "继续你的任务。"
                } else {
                    &task.resume_note
                }
            )];
            if let Some(cp) = task.last_checkpoint() {
                parts.push(format!(
                    "你上次的打点：{}{}",
                    cp.phase,
                    if cp.summary.is_empty() {
                        String::new()
                    } else {
                        format!(" — {}", cp.summary)
                    }
                ));
            }
            if !inbox.is_empty() {
                parts.push(format!("主 agent / 用户补充：\n{inbox}"));
            }
            return parts.join("\n\n");
        }
        let profile = resolve_profile(&self.profiles, &task.profile);
        let mut parts = vec![
            format!("任务 {}：{}", task.id, task.title),
            task.brief.trim().to_string(),
        ];
        if self.inherits_prompt(task, &profile) {
            let names = profile.tool_names().unwrap_or_default().join(", ");
            let role = fork_child_prompt(
                task.agent_name(),
                &task.id,
                &task.title,
                if names.is_empty() {
                    "（主会话的全部工具）"
                } else {
                    &names
                },
            );
            parts.insert(0, profile.body.clone());
            parts.insert(0, role);
        }
        if !task.acceptance.is_empty() {
            parts.push(format!("验收标准：{}", task.acceptance));
        }
        if !task.scope_paths.is_empty() {
            parts.push(format!("你认领的路径：{}", task.scope_paths.join(", ")));
        }
        if !task.resume_note.is_empty() {
            parts.push(format!("前置信息：\n{}", task.resume_note));
        }
        if !inbox.is_empty() {
            parts.push(format!("主 agent / 用户补充：\n{inbox}"));
        }
        parts.retain(|p| !p.is_empty());
        parts.join("\n\n")
    }

    async fn run(
        &self,
        id: &str,
        cancel: CancellationToken,
        steer: Arc<SteerQueue>,
        live_messages: Arc<Mutex<Vec<Message>>>,
    ) -> anyhow::Result<()> {
        let mut task = self
            .board
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no task {id}"))?;
        if task.isolation == "worktree"
            && task
                .worktree
                .as_ref()
                .is_none_or(|w| !Path::new(w).is_dir())
        {
            let wd = self.workdir.clone();
            let (sid, tid) = (self.main_id.clone(), task.id.clone());
            let tree =
                tokio::task::spawn_blocking(move || worktree::create(&wd, &sid, &tid)).await??;
            self.update(|b| {
                if let Some(t) = b.get_mut(id) {
                    t.worktree = Some(tree.0.display().to_string());
                    t.branch = Some(tree.1.clone());
                }
            });
            task.worktree = Some(tree.0.display().to_string());
            task.branch = Some(tree.1);
        }
        let (mut agent, resumed) = self.open_agent(&task, steer)?;
        let session_file = agent.session_path().display().to_string();
        let mut text = self.first_input(&task, resumed);
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                t.session_file = Some(session_file);
                t.resume_note.clear();
                t.inbox.clear();
            }
        });
        let budget = task.max_tokens;
        let every = self.config.subagents.auto_checkpoint_every;
        let mut last_text = String::new();
        let mut last_error = String::new();
        let mut writes: Vec<String> = Vec::new();
        let mut over_budget = String::new();
        loop {
            let (tx, mut rx) = mpsc::unbounded_channel();
            let mut calls = 0usize;
            {
                let run = agent.run(std::mem::take(&mut text), vec![], &tx, &cancel);
                tokio::pin!(run);
                loop {
                    tokio::select! {
                        _ = &mut run => break,
                        Some(ev) = rx.recv() => {
                            match &ev {
                                AgentEvent::ToolCall { name, .. } => {
                                    calls += 1;
                                    self.set_activity(id, &format!("⚙ {name}"));
                                    if every > 0 && calls.is_multiple_of(every) {
                                        let summary: String = last_text.chars().rev().take(200).collect::<Vec<_>>().into_iter().rev().collect();
                                        self.add_checkpoint(id, &format!("已执行 {calls} 次工具调用"), &summary, vec![], None, true);
                                    }
                                }
                                AgentEvent::Text(t) => {
                                    last_text.push_str(t);
                                    if last_text.len() > 4000 {
                                        last_text = last_text.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
                                    }
                                    self.set_activity(id, "✎ 回复中");
                                }
                                AgentEvent::Error(e) => last_error = e.clone(),
                                AgentEvent::ToolResult { name, success: true, output, .. }
                                    if !task_tools::SUBAGENT_TOOL_NAMES.contains(&name.as_str())
                                        && !["read", "grep", "find", "ls", "session_search", "web_fetch", "web_search"].contains(&name.as_str()) =>
                                {
                                    let first: String = output.trim().lines().next().unwrap_or("完成").chars().take(160).collect();
                                    writes.push(format!("{name}：{first}"));
                                }
                                AgentEvent::Usage { session_total, .. }
                                    if budget > 0 && *session_total > budget && over_budget.is_empty() =>
                                {
                                    over_budget = format!("超出 token 预算（{session_total} > {budget}）");
                                    cancel.cancel();
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            while rx.try_recv().is_ok() {}
            *live_messages.lock().unwrap() = agent.messages.clone();
            let tokens = agent.session_total_tokens();
            self.update(|b| {
                if let Some(t) = b.get_mut(id) {
                    t.tokens = tokens;
                }
            });
            if !over_budget.is_empty() {
                let terminal = self
                    .board
                    .lock()
                    .unwrap()
                    .get(id)
                    .is_none_or(|t| t.terminal());
                if !terminal {
                    let mut reason = vec![format!("{over_budget}，已收尾，结果可能不完整")];
                    if !writes.is_empty() {
                        let tail: Vec<String> = writes
                            .iter()
                            .rev()
                            .take(20)
                            .rev()
                            .map(|w| format!("- {w}"))
                            .collect();
                        reason.push(format!("已执行的改动：\n{}", tail.join("\n")));
                    }
                    let summary = last_answer(&agent.messages);
                    let failed = summary.is_empty() && writes.is_empty();
                    self.finish(id, &summary, vec![], vec![], failed, &reason.join("\n"));
                }
                break;
            }
            let status = self
                .board
                .lock()
                .unwrap()
                .get(id)
                .map(|t| t.status.clone())
                .unwrap_or_default();
            if status != "running" || agent.was_interrupted || cancel.is_cancelled() {
                break;
            }
            let leftovers = agent.steer.drain_modes(&[STEER, FOLLOW_UP]);
            if !leftovers.is_empty() {
                text = leftovers
                    .iter()
                    .map(|i| i.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                continue;
            }
            if agent.last_run_failed {
                let reason = if last_error.is_empty() {
                    "LLM 调用失败".to_string()
                } else {
                    last_error.clone()
                };
                self.finish(id, &last_text, vec![], vec![], true, &reason);
                break;
            }
            // The turn ended without finish(): its last answer is the result.
            let answer = last_answer(&agent.messages);
            let answer = if answer.is_empty() {
                "（子代理没有给出总结）".to_string()
            } else {
                answer
            };
            self.finish(id, &answer, vec![], vec![], false, "");
            break;
        }
        agent.checkpoint();
        // A finished worktree task commits its changes on its branch.
        let wt = self
            .board
            .lock()
            .unwrap()
            .get(id)
            .filter(|t| t.status == "done")
            .and_then(|t| t.worktree.clone());
        if let Some(path) = wt {
            let msg = format!("limbo {id}: {}", task.title);
            let _ =
                tokio::task::spawn_blocking(move || worktree::commit_all(Path::new(&path), &msg))
                    .await;
        }
        Ok(())
    }

    fn set_activity(&self, id: &str, text: &str) {
        let mut a = self.activity.lock().unwrap();
        if a.get(id).map(String::as_str) != Some(text) {
            a.insert(id.into(), text.into());
        }
    }

    pub fn activity(&self, id: &str) -> String {
        self.activity
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .unwrap_or_default()
    }

    // -- notices ---------------------------------------------------------------------

    fn notice(&self, task_id: &str, kind: &str, text: &str, headline: &str, nid: Option<String>) {
        let n = Notice {
            id: nid.unwrap_or_else(|| {
                format!(
                    "{}:{kind}:{}",
                    if task_id.is_empty() { "board" } else { task_id },
                    uuid_like()
                )
            }),
            kind: kind.into(),
            text: text.into(),
            headline: if headline.is_empty() {
                text.lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(160)
                    .collect()
            } else {
                headline.into()
            },
        };
        background::emit_for(
            &self.main_id,
            BackgroundEvent::Notice {
                kind: kind.into(),
                text: n.headline.clone(),
            },
        );
        if DELIVERED.contains(&kind) {
            self.pending.lock().unwrap().push(n);
        }
    }

    fn result_notice(&self, t: &Task) -> Option<Notice> {
        let oc = t.outcome.as_ref()?;
        let kind = match oc.status.as_str() {
            "done" => "result",
            "failed" => "alert",
            _ => "info",
        };
        let limit = self.config.subagents.result_chars;
        let mut body = oc.summary.trim().to_string();
        if body.chars().count() > limit {
            body = body.chars().take(limit).collect::<String>() + "…（已截断，完整内容见记录）";
        }
        let mut lines = vec![];
        if !body.is_empty() {
            lines.push(body.clone());
        }
        if !oc.reason.is_empty() {
            lines.push(format!("原因：{}", oc.reason));
        }
        if !oc.files_changed.is_empty() {
            let f: Vec<&str> = oc
                .files_changed
                .iter()
                .take(30)
                .map(String::as_str)
                .collect();
            lines.push(format!("改动文件：{}", f.join(", ")));
        }
        if !oc.follow_ups.is_empty() {
            let f: Vec<&str> = oc.follow_ups.iter().take(10).map(String::as_str).collect();
            lines.push(format!("后续建议：{}", f.join("；")));
        }
        if let Some(w) = &t.worktree {
            lines.push(format!(
                "worktree：{w}（合并：task_control(id=\"{}\", action=\"merge\")）",
                t.id
            ));
        }
        if let Some(s) = &t.session_file {
            lines.push(format!("完整记录：{s}"));
        }
        let text = format!(
            "<subagent_update task=\"{}\" agent=\"{}\" kind=\"{kind}\" status=\"{}\">\n{}\n</subagent_update>",
            t.id,
            t.agent_name(),
            oc.status,
            lines.join("\n")
        );
        let first = if oc.reason.is_empty() {
            &body
        } else {
            &oc.reason
        };
        let first: String = first
            .lines()
            .next()
            .unwrap_or(&t.title)
            .chars()
            .take(120)
            .collect();
        Some(Notice {
            id: format!("{}:result:{}", t.id, t.attempt),
            kind: kind.into(),
            text,
            headline: format!(
                "{} {} {}：{first}",
                t.id,
                t.agent_name(),
                status_label(&oc.status)
            ),
        })
    }

    fn question_notice(t: &Task, q: &Question) -> Notice {
        Notice {
            id: q.id.clone(),
            kind: "question".into(),
            text: format!(
                "<subagent_update task=\"{}\" agent=\"{}\" kind=\"question\" id=\"{}\">\n{}\n（回答：answer(question_id=\"{}\", text=...)；需要用户决定时先问用户）\n</subagent_update>",
                t.id,
                t.agent_name(),
                q.id,
                q.text,
                q.id
            ),
            headline: format!(
                "{} {} 提问：{}",
                t.id,
                t.agent_name(),
                q.text.chars().take(120).collect::<String>()
            ),
        }
    }

    fn regenerate_notices(&self) {
        let b = self.board.lock().unwrap();
        let mut pending = self.pending.lock().unwrap();
        for t in &b.tasks {
            if t.terminal()
                && t.created_by != "user"
                && t.status != "cancelled"
                && let Some(n) = self.result_notice(t)
                && !b.reported.contains(&n.id)
            {
                pending.push(n);
            }
            if let Some(q) = t.open_question()
                && !b.reported.contains(&q.id)
            {
                pending.push(Self::question_notice(t, q));
            }
        }
    }

    fn deliver(&self) {
        let deliverable: Vec<Notice> = {
            let b = self.board.lock().unwrap();
            let mut p = self.pending.lock().unwrap();
            p.retain(|n| DELIVERED.contains(&n.kind.as_str()) && !b.reported.contains(&n.id));
            p.clone()
        };
        if deliverable.is_empty() {
            return;
        }
        if self.main_busy() {
            return self.queue(deliverable, false);
        }
        let wake: HashSet<&str> = self
            .config
            .subagents
            .wake_main_on
            .iter()
            .map(String::as_str)
            .collect();
        let kinds: HashSet<&str> = deliverable.iter().map(|n| n.kind.as_str()).collect();
        let urgent = ["question", "alert"]
            .iter()
            .any(|k| kinds.contains(k) && wake.contains(k));
        let wave_done = self
            .board
            .lock()
            .unwrap()
            .with_status(&["running", "ready"])
            .is_empty();
        let results = kinds.contains("result")
            && (wake.contains("result") || (wake.contains("wave_done") && wave_done));
        if urgent || results {
            self.queue(deliverable, true);
        }
    }

    /// Queue every undelivered notice now (before a user turn starts, so the
    /// main agent sees them ahead of the user's message).
    pub fn flush_notices(&self) {
        let deliverable: Vec<Notice> = {
            let b = self.board.lock().unwrap();
            self.pending
                .lock()
                .unwrap()
                .iter()
                .filter(|n| DELIVERED.contains(&n.kind.as_str()) && !b.reported.contains(&n.id))
                .cloned()
                .collect()
        };
        if !deliverable.is_empty() {
            self.queue(deliverable, false);
        }
    }

    fn queue(&self, notices: Vec<Notice>, wake: bool) {
        let ids: Vec<String> = notices.iter().map(|n| n.id.clone()).collect();
        let board_line = {
            let b = self.board.lock().unwrap();
            b.live()
                .iter()
                .map(|t| format!("{} {}", t.id, t.status))
                .collect::<Vec<_>>()
                .join(" · ")
        };
        let text = format!(
            "<subagent_updates board=\"{}\">\n{}\n</subagent_updates>",
            if board_line.is_empty() {
                "no live tasks"
            } else {
                &board_line
            },
            notices
                .iter()
                .map(|n| n.text.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
        self.main_steer
            .push_item(text, vec![], STEER, "subagent", ids.clone());
        self.update(|b| b.reported.extend(ids.iter().cloned()));
        self.pending
            .lock()
            .unwrap()
            .retain(|n| !ids.contains(&n.id));
        if wake {
            background::emit_for(&self.main_id, BackgroundEvent::MainWake);
        }
    }

    /// Drain the notices queued for the idle main agent into one input.
    pub fn take_wake_input(&self) -> Option<String> {
        if !self
            .main_steer
            .items()
            .iter()
            .any(|i| i.source == "subagent")
        {
            return None;
        }
        let drained = self.main_steer.drain_modes(&[STEER]);
        let ids: Vec<String> = drained.iter().map(|i| i.id.clone()).collect();
        self.main_steer.journal_dequeued(&ids);
        let text = drained
            .into_iter()
            .map(|i| i.text)
            .collect::<Vec<_>>()
            .join("\n\n");
        (!text.is_empty()).then_some(text)
    }

    // -- verbs: main agent / user -------------------------------------------------------

    pub fn plan(&self, specs: &[Value], created_by: &str) -> PlanResult {
        let fail = |m: String| PlanResult {
            message: m,
            proposed: false,
            ids: vec![],
        };
        let s = &self.config.subagents;
        let live = self.board.lock().unwrap().live().len();
        if live + specs.len() > s.max_tasks {
            return fail(format!(
                "超出任务上限：最多 {} 个未结束任务，当前 {live} 个。合并任务或等现有任务结束。",
                s.max_tasks
            ));
        }
        let ordered = match topo(specs) {
            Ok(o) => o,
            Err(e) => return fail(e),
        };
        let status =
            if (s.auto == "suggest" && created_by != "user") || created_by.ends_with("-proposal") {
                "proposed"
            } else {
                "pending"
            };
        let mut is_git: Option<bool> = None;
        let result = self.update(|b| {
            let mut keys: HashMap<String, String> = HashMap::new();
            let mut created: Vec<Task> = Vec::new();
            for spec in &ordered {
                let mut deps = Vec::new();
                for d in str_list(spec.get("depends_on")) {
                    if let Some(id) = keys.get(&d) {
                        deps.push(id.clone());
                    } else if let Some(t) = b.find(&d) {
                        deps.push(t.id.clone());
                    } else {
                        return Err(format!("依赖 {d} 既不是本计划的 key，也不是已有任务"));
                    }
                }
                let profile = resolve_profile(
                    &self.profiles,
                    spec.get("profile")
                        .and_then(Value::as_str)
                        .unwrap_or("worker"),
                );
                let mut isolation = spec
                    .get("isolation")
                    .and_then(Value::as_str)
                    .map(String::from)
                    .unwrap_or_else(|| {
                        if profile.isolation != "shared" {
                            profile.isolation.clone()
                        } else {
                            s.default_isolation.clone()
                        }
                    });
                if isolation == "worktree"
                    && !*is_git.get_or_insert_with(|| worktree::is_git_repo(&self.workdir))
                {
                    isolation = "shared".into();
                }
                let mut scope = str_list(spec.get("scope_paths"));
                if scope.is_empty() {
                    scope = profile.scope_paths.clone();
                }
                b.next_id += 1;
                let t = Task {
                    id: format!("t{}", b.next_id),
                    title: spec
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or("untitled")
                        .into(),
                    brief: spec
                        .get("brief")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .into(),
                    profile: profile.name.clone(),
                    status: status.into(),
                    depends_on: deps,
                    created_by: created_by.into(),
                    acceptance: spec
                        .get("acceptance")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .into(),
                    scope_paths: scope,
                    priority: spec.get("priority").and_then(Value::as_i64).unwrap_or(0),
                    isolation,
                    context: if spec.get("context").and_then(Value::as_str) == Some("fork")
                        || profile.context == "fork"
                    {
                        "fork".into()
                    } else {
                        "brief".into()
                    },
                    on_dep_fail: s.on_dep_fail.clone(),
                    max_tokens: s.task_token_budget,
                    ..Default::default()
                };
                if let Some(k) = spec.get("key").and_then(Value::as_str) {
                    keys.insert(k.into(), t.id.clone());
                }
                created.push(t.clone());
                b.tasks.push(t);
            }
            Ok(created)
        });
        let created = match result {
            Ok(c) => c,
            Err(e) => return fail(e),
        };
        let listing = created
            .iter()
            .map(|t| {
                format!(
                    "- {} [{}] {}{}",
                    t.id,
                    t.profile,
                    t.title,
                    if t.depends_on.is_empty() {
                        String::new()
                    } else {
                        format!("（依赖 {}）", t.depends_on.join(", "))
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let ids: Vec<String> = created.iter().map(|t| t.id.clone()).collect();
        if status == "proposed" {
            background::emit_for(&self.main_id, BackgroundEvent::PlanProposed(ids.clone()));
            return PlanResult {
                message: format!(
                    "计划已提交给用户确认：\n{listing}\n用户确认前任务不会开始。现在结束本轮，简短告诉用户计划的要点即可。"
                ),
                proposed: true,
                ids,
            };
        }
        self.reconcile();
        PlanResult {
            message: format!(
                "已创建 {} 个任务并在后台开始：\n{listing}\n结果会以 <subagent_updates> 回到你这里；现在可以先回复用户。",
                created.len()
            ),
            proposed: false,
            ids,
        }
    }

    fn resolve(&self, r: &str, statuses: Option<&[&str]>) -> Result<Vec<String>, String> {
        let b = self.board.lock().unwrap();
        if r.trim() == "all" {
            return Ok(b
                .live()
                .into_iter()
                .filter(|t| statuses.is_none_or(|s| s.contains(&t.status.as_str())))
                .map(|t| t.id.clone())
                .collect());
        }
        b.find(r)
            .map(|t| vec![t.id.clone()])
            .ok_or_else(|| format!("没有任务或子代理 {r}"))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn control(
        &self,
        r: &str,
        action: &str,
        note: &str,
        priority: Option<i64>,
        fresh: bool,
        brief: Option<String>,
        depends_on: Option<Vec<String>>,
        by: &str,
    ) -> String {
        let statuses: Option<&[&str]> = match action {
            "pause" => Some(&["running", "waiting", "ready", "pending"]),
            "resume" => Some(&["paused"]),
            "approve" | "reject" => Some(&["proposed"]),
            _ => None,
        };
        let ids = match self.resolve(r, statuses) {
            Ok(ids) => ids,
            Err(e) => return e,
        };
        if ids.is_empty() {
            return format!("没有可以 {action} 的任务");
        }
        let mut out = Vec::new();
        for id in ids {
            let line = match action {
                "pause" => self.pause(&id, note),
                "resume" => self.resume(&id, note),
                "cancel" => {
                    let reason = if !note.is_empty() {
                        note.to_string()
                    } else if by == "user" {
                        "用户取消".into()
                    } else {
                        "主 agent 取消".into()
                    };
                    self.cancel_task(&id, &reason, by)
                }
                "retry" => self.retry(&id, note, fresh),
                "reprioritize" => {
                    let p = priority.unwrap_or(0);
                    self.update(|b| {
                        if let Some(t) = b.get_mut(&id) {
                            t.priority = p;
                        }
                    });
                    format!("{id} 优先级 → {p}")
                }
                "edit" => self.edit(&id, brief.clone(), depends_on.clone(), note),
                "approve" => {
                    self.update(|b| {
                        if let Some(t) = b.get_mut(&id) {
                            t.status = "pending".into();
                        }
                    });
                    format!("{id} 已批准")
                }
                "reject" => {
                    self.set_outcome(
                        &id,
                        "cancelled",
                        "",
                        if note.is_empty() {
                            "用户拒绝"
                        } else {
                            note
                        },
                    );
                    format!("{id} 已拒绝")
                }
                "merge" => self.merge(&id),
                other => format!("未知动作 {other}"),
            };
            out.push(line);
        }
        self.reconcile();
        out.join("\n")
    }

    fn pause(&self, id: &str, note: &str) -> String {
        if let Some(r) = self.runners.lock().unwrap().get(id) {
            r.cancel.cancel();
        }
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                t.paused_from = t.status.clone();
                t.status = "paused".into();
                if !note.is_empty() {
                    t.resume_note = join_notes(&t.resume_note, note);
                }
            }
        });
        format!("{id} 已暂停")
    }

    fn resume(&self, id: &str, note: &str) -> String {
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                t.status = if t.paused_from == "pending" || t.paused_from == "waiting" {
                    std::mem::take(&mut t.paused_from)
                } else {
                    "ready".into()
                };
                t.resume_note = join_notes(&t.resume_note, note);
            }
        });
        format!("{id} 已恢复")
    }

    fn set_outcome(&self, id: &str, status: &str, summary: &str, reason: &str) {
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                t.status = status.into();
                t.outcome = Some(Outcome {
                    status: status.into(),
                    summary: summary.into(),
                    reason: reason.into(),
                    ..Default::default()
                });
            }
        });
    }

    pub fn cancel_task(&self, id: &str, reason: &str, by: &str) -> String {
        let terminal = self
            .board
            .lock()
            .unwrap()
            .get(id)
            .is_none_or(|t| t.terminal());
        if terminal {
            return format!("{id} 已经结束");
        }
        if let Some(r) = self.runners.lock().unwrap().get(id) {
            r.cancel.cancel();
        }
        self.set_outcome(id, "cancelled", "", reason);
        if by == "system" {
            self.notice(
                id,
                "alert",
                &format!("任务 {id} 已取消：{reason}"),
                "",
                None,
            );
        }
        format!("{id} 已取消")
    }

    fn retry(&self, id: &str, note: &str, fresh: bool) -> String {
        if self.runners.lock().unwrap().contains_key(id) {
            return format!("{id} 正在运行，先 cancel 再 retry");
        }
        let fresh_file = fresh.then(uuid_like);
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                t.status = "ready".into();
                t.outcome = None;
                t.attempt += 1;
                t.dep_alerted = false;
                t.resume_note = join_notes(&t.resume_note, note);
                if let Some(suffix) = &fresh_file {
                    let dir = self.agents_dir();
                    t.session_file = Some(
                        dir.join(format!("{}-{}-{suffix}.jsonl", t.id, t.profile))
                            .display()
                            .to_string(),
                    );
                }
            }
        });
        format!("{id} 将重试{}", if fresh { "（全新记录）" } else { "" })
    }

    fn edit(
        &self,
        id: &str,
        brief: Option<String>,
        deps: Option<Vec<String>>,
        note: &str,
    ) -> String {
        let running = self
            .runners
            .lock()
            .unwrap()
            .get(id)
            .map(|r| r.steer.clone());
        if let Some(steer) = running {
            let text = brief.clone().unwrap_or_else(|| note.to_string());
            if text.is_empty() {
                return format!("{id} 正在运行：edit 需要 brief 或 note 作为更正");
            }
            steer.push_item(
                format!("（主 agent 更正任务）{text}"),
                vec![],
                STEER,
                "user",
                vec![],
            );
            return format!("{id} 正在运行，更正已送达");
        }
        self.update(|b| {
            let Some(t) = b.get_mut(id) else {
                return format!("没有任务 {id}");
            };
            if t.session_file.is_some() && t.status != "proposed" {
                if let Some(text) = brief
                    .clone()
                    .or(Some(note.to_string()).filter(|s| !s.is_empty()))
                {
                    t.inbox.push(text);
                }
                return format!("{id} 已开始过：更正会在它下次运行时送达");
            }
            if let Some(br) = brief {
                t.brief = br;
            }
            if let Some(d) = deps {
                t.depends_on = d;
                t.dep_alerted = false;
                if t.status == "waiting" {
                    t.status = "pending".into();
                }
            }
            format!("{id} 已修改")
        })
    }

    fn merge(&self, id: &str) -> String {
        let (status, branch) = {
            let b = self.board.lock().unwrap();
            match b.get(id) {
                Some(t) => (t.status.clone(), t.branch.clone()),
                None => return format!("没有任务 {id}"),
            }
        };
        let Some(branch) = branch else {
            return format!("{id} 不是 worktree 任务");
        };
        if status != "done" {
            return format!("{id} 还没完成（{}）", status_label(&status));
        }
        match worktree::merge(&self.workdir, &branch) {
            (true, msg) => format!("{id} 已合并 {branch}：{msg}"),
            (false, msg) => format!("{id} 合并失败（已中止）：{msg}"),
        }
    }

    /// A message to a subagent (running: steered in; otherwise its inbox).
    pub fn send(&self, r: &str, text: &str, mode: &str, by: &str) -> String {
        let id = match self.resolve(r, None) {
            Ok(ids) if ids.len() == 1 => ids[0].clone(),
            Ok(_) => return "send 需要一个具体的任务".into(),
            Err(e) => return e,
        };
        let who = if by == "user" { "用户" } else { "主 agent" };
        let text = format!("（{who}）{text}");
        if let Some(r) = self.runners.lock().unwrap().get(&id) {
            let m = if mode == FOLLOW_UP { FOLLOW_UP } else { STEER };
            r.steer.push_item(text, vec![], m, "user", vec![]);
            return format!("已发给 {id}");
        }
        self.update(|b| {
            if let Some(t) = b.get_mut(&id) {
                t.inbox.push(text);
            }
        });
        format!("{id} 没在运行：消息会在它开始/恢复时送达")
    }

    pub fn answer(&self, qid: &str, text: &str) -> String {
        let found = self.update(|b| {
            for t in b.tasks.iter_mut() {
                if let Some(q) = t.questions.iter_mut().find(|q| q.id == qid) {
                    q.answer = Some(text.to_string());
                    t.inbox.push(format!("问题「{}」的回答：{text}", q.text));
                    if t.status == "waiting" && t.waiting_on.is_empty() {
                        t.status = "ready".into();
                        t.wait_reason.clear();
                    }
                    return Some(t.id.clone());
                }
            }
            None
        });
        match found {
            Some(id) => {
                self.reconcile();
                format!("已回答，{id} 将继续")
            }
            None => format!("没有问题 {qid}"),
        }
    }

    // -- verbs: subagents ----------------------------------------------------------

    pub fn add_checkpoint(
        &self,
        id: &str,
        phase: &str,
        summary: &str,
        files: Vec<String>,
        progress: Option<f64>,
        auto: bool,
    ) {
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                let seq = t.checkpoints.len() as u32 + 1;
                t.checkpoints.push(Checkpoint {
                    seq,
                    phase: phase.into(),
                    summary: summary.into(),
                    files,
                    progress,
                    auto,
                });
            }
        });
    }

    pub fn finish(
        &self,
        id: &str,
        summary: &str,
        files: Vec<String>,
        follow_ups: Vec<String>,
        failed: bool,
        reason: &str,
    ) -> String {
        let status = if failed { "failed" } else { "done" };
        let task = self.update(|b| {
            let t = b.get_mut(id)?;
            if t.terminal() {
                return None;
            }
            t.status = status.into();
            t.outcome = Some(Outcome {
                status: status.into(),
                summary: summary.into(),
                reason: reason.into(),
                files_changed: files,
                follow_ups,
            });
            Some(t.clone())
        });
        if let Some(t) = task
            && t.created_by != "user"
            && let Some(n) = self.result_notice(&t)
        {
            background::emit_for(
                &self.main_id,
                BackgroundEvent::Notice {
                    kind: n.kind.clone(),
                    text: n.headline.clone(),
                },
            );
            self.pending.lock().unwrap().push(n);
        } else if let Some(t) = self.board.lock().unwrap().get(id).cloned()
            && t.created_by == "user"
            && let Some(n) = self.result_notice(&t)
        {
            // User-started tasks (e.g. /learn) report to the user only.
            background::emit_for(
                &self.main_id,
                BackgroundEvent::Notice {
                    kind: n.kind,
                    text: n.headline,
                },
            );
        }
        "已交付，任务结束。".into()
    }

    pub fn child_ask(&self, id: &str, question: &str) -> String {
        let q = self.update(|b| {
            let t = b.get_mut(id)?;
            let q = Question {
                id: format!("{id}-q{}", t.questions.len() + 1),
                text: question.into(),
                answer: None,
            };
            t.questions.push(q.clone());
            t.status = "waiting".into();
            t.wait_reason = "等待回答".into();
            Some((t.clone(), q))
        });
        if let Some((t, q)) = q {
            let n = Self::question_notice(&t, &q);
            background::emit_for(
                &self.main_id,
                BackgroundEvent::Notice {
                    kind: "question".into(),
                    text: n.headline.clone(),
                },
            );
            self.pending.lock().unwrap().push(n);
        }
        "问题已提交，你会在得到回答后继续。现在结束本轮。".into()
    }

    pub fn child_need(
        &self,
        id: &str,
        on: Vec<String>,
        new: Option<&Value>,
        reason: &str,
        policy: &str,
    ) -> String {
        if policy == "cancel_self" {
            self.set_outcome(
                id,
                "cancelled",
                "",
                &format!("子代理判断任务已过时：{reason}"),
            );
            return "任务已取消。".into();
        }
        if policy == "ask_main" {
            return self.child_ask(id, &format!("依赖问题：{reason}"));
        }
        let mut waits = Vec::new();
        for r in on {
            match self.board.lock().unwrap().find(&r) {
                Some(t) => waits.push(t.id.clone()),
                None => return format!("没有任务 {r}"),
            }
        }
        if let Some(spec) = new {
            let mut spec = spec.clone();
            if spec.get("brief").is_none() {
                spec["brief"] = json!(reason);
            }
            let r = self.plan(&[spec], id);
            waits.extend(r.ids);
        }
        if waits.is_empty() {
            return "need 需要 on 或 new".into();
        }
        let list = waits.join(", ");
        self.update(|b| {
            if let Some(t) = b.get_mut(id) {
                t.status = "waiting".into();
                t.waiting_on = waits;
                t.wait_reason = reason.into();
            }
        });
        format!("已登记等待 {list}；完成后你会带着它们的结果继续。现在结束本轮。")
    }

    pub fn child_propose(&self, id: &str, title: &str, brief: &str, profile: &str) -> String {
        let r = self.plan(
            &[json!({"title": title, "brief": brief, "profile": profile})],
            &format!("{id}-proposal"),
        );
        self.notice(id, "info", &format!("{id} 提议新任务：{title}"), "", None);
        format!("已提议：{}", r.ids.join(", "))
    }

    // -- views ---------------------------------------------------------------------

    pub fn tasks(&self) -> Vec<Task> {
        self.board.lock().unwrap().tasks.clone()
    }

    pub fn find(&self, r: &str) -> Option<Task> {
        self.board.lock().unwrap().find(r).cloned()
    }

    pub fn working(&self) -> bool {
        !self.runners.lock().unwrap().is_empty()
            || !self
                .board
                .lock()
                .unwrap()
                .with_status(&["ready", "running"])
                .is_empty()
    }

    pub fn summary_counts(&self) -> Vec<(String, usize)> {
        let b = self.board.lock().unwrap();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for t in b.live() {
            *counts.entry(t.status.clone()).or_default() += 1;
        }
        counts.into_iter().collect()
    }

    pub fn counts_label(&self) -> String {
        self.summary_counts()
            .iter()
            .map(|(s, n)| format!("{n} {}", status_label(s)))
            .collect::<Vec<_>>()
            .join(" · ")
    }

    pub fn status_text(&self, r: Option<&str>) -> String {
        let b = self.board.lock().unwrap();
        if let Some(r) = r {
            return match b.find(r) {
                Some(t) => self.detail(t),
                None => format!("没有任务或子代理 {r}"),
            };
        }
        if b.tasks.is_empty() {
            return "任务板是空的。".into();
        }
        b.tasks
            .iter()
            .map(|t| {
                let mut line = format!(
                    "{} [{}] {} · {}",
                    t.id,
                    status_label(&t.status),
                    t.agent_name(),
                    t.title
                );
                if !t.depends_on.is_empty() && (t.status == "pending" || t.status == "proposed") {
                    line += &format!("（依赖 {}）", t.depends_on.join(", "));
                }
                if t.status == "waiting" {
                    line += &format!(
                        "（{}）",
                        if t.wait_reason.is_empty() {
                            format!("等待 {}", t.waiting_on.join(", "))
                        } else {
                            t.wait_reason.clone()
                        }
                    );
                }
                if let Some(cp) = t.last_checkpoint()
                    && !t.terminal()
                {
                    line += &format!(" · 打点：{}", cp.phase);
                }
                let act = self.activity(&t.id);
                if t.status == "running" && !act.is_empty() {
                    line += &format!(" · {act}");
                }
                if let Some(o) = &t.outcome
                    && t.terminal()
                {
                    let head = if o.reason.is_empty() {
                        &o.summary
                    } else {
                        &o.reason
                    };
                    if let Some(h) = head.trim().lines().next() {
                        line += &format!(" · {}", h.chars().take(100).collect::<String>());
                    }
                }
                line
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn detail(&self, t: &Task) -> String {
        let mut lines = vec![
            format!("{} {}", t.id, t.title),
            format!(
                "状态：{}；角色：{}；子代理：{}",
                status_label(&t.status),
                t.profile,
                t.agent_name()
            ),
        ];
        if !t.depends_on.is_empty() {
            lines.push(format!("依赖：{}", t.depends_on.join(", ")));
        }
        if !t.waiting_on.is_empty() {
            lines.push(format!(
                "等待：{}（{}）",
                t.waiting_on.join(", "),
                t.wait_reason
            ));
        }
        if !t.scope_paths.is_empty() {
            lines.push(format!("认领路径：{}", t.scope_paths.join(", ")));
        }
        lines.push(format!(
            "说明：{}",
            t.brief.chars().take(1500).collect::<String>()
        ));
        for cp in t.checkpoints.iter().rev().take(5).rev() {
            lines.push(format!(
                "打点 {}：{}{}",
                cp.seq,
                cp.phase,
                if cp.summary.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", cp.summary.chars().take(200).collect::<String>())
                }
            ));
        }
        for q in &t.questions {
            lines.push(format!(
                "问题 {}：{}{}",
                q.id,
                q.text,
                q.answer
                    .as_ref()
                    .map_or("（未回答）".to_string(), |a| format!(" → {a}"))
            ));
        }
        if let Some(o) = &t.outcome {
            lines.push(format!(
                "结果：{} {}\n{}",
                o.status,
                o.reason,
                o.summary.chars().take(2000).collect::<String>()
            ));
        }
        if let Some(s) = &t.session_file {
            lines.push(format!("记录：{s}"));
        }
        if t.tokens > 0 {
            lines.push(format!("tokens：{}", t.tokens));
        }
        lines.join("\n")
    }

    /// (role, text) pairs of a subagent's history for the UI viewer.
    pub fn transcript(&self, r: &str, limit: usize) -> Vec<(String, String)> {
        let Some(t) = self.find(r) else { return vec![] };
        let live = self
            .runners
            .lock()
            .unwrap()
            .get(&t.id)
            .map(|r| r.messages.lock().unwrap().clone());
        let messages = match live.filter(|m| !m.is_empty()) {
            Some(m) => m,
            None => match t
                .session_file
                .as_ref()
                .and_then(|p| session::load_session_full(Path::new(p)).ok())
            {
                Some(l) => l.messages,
                None => return vec![],
            },
        };
        let mut out = Vec::new();
        for m in messages.iter().skip(1) {
            match m.role {
                Role::Tool => out.push(("tool".into(), m.text().chars().take(300).collect())),
                Role::Assistant => {
                    if !m.text().is_empty() {
                        out.push(("assistant".into(), m.text().to_string()));
                    }
                    for tc in m.tool_calls.iter().flatten() {
                        out.push(("call".into(), tc.name().to_string()));
                    }
                }
                Role::User => out.push(("user".into(), m.text().to_string())),
                Role::System => {}
            }
        }
        let skip = out.len().saturating_sub(limit);
        out.split_off(skip)
    }

    /// Stop every subagent without writing outcomes (recovered on next load).
    pub fn stop(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return;
        }
        for r in self.runners.lock().unwrap().values() {
            r.cancel.cancel();
        }
    }

    /// Whether a ClaimPolicy should block `rel` for task `me`.
    pub fn claimed_by(&self, me: &str, rel: &str) -> Option<(String, String, String)> {
        let b = self.board.lock().unwrap();
        for t in b.live() {
            if t.id == me || t.isolation == "worktree" {
                continue;
            }
            if !t.scope_paths.is_empty() && super::policies::matches(rel, &t.scope_paths) {
                return Some((t.id.clone(), t.agent_name().to_string(), t.title.clone()));
            }
        }
        None
    }

    /// Learn: whether a skill review is already live.
    pub fn review_running(&self) -> bool {
        self.board
            .lock()
            .unwrap()
            .live()
            .iter()
            .any(|t| t.profile == "skill-reviewer")
    }

    pub fn outcome_of(&self, id: &str) -> Option<Outcome> {
        self.board
            .lock()
            .unwrap()
            .get(id)
            .and_then(|t| t.outcome.clone())
    }
}

fn path_stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::SeqCst);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}{n:x}", t & 0xffff_ffff)
}

/// A JSON value as a list of strings (array, JSON text, or comma list).
pub fn str_list(v: Option<&Value>) -> Vec<String> {
    match v {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(String::from)
                    .unwrap_or_else(|| x.to_string())
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(a)) => str_list(Some(&Value::Array(a))),
            _ => s
                .split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect(),
        },
        Some(other) => vec![other.to_string()],
    }
}

/// Order plan entries so every in-plan dependency comes first.
fn topo(specs: &[Value]) -> Result<Vec<Value>, String> {
    let by_key: HashMap<String, usize> = specs
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            s.get("key")
                .and_then(Value::as_str)
                .map(|k| (k.to_string(), i))
        })
        .collect();
    let mut done = vec![false; specs.len()];
    let mut visiting = vec![false; specs.len()];
    let mut out = Vec::new();
    fn visit(
        i: usize,
        specs: &[Value],
        by_key: &HashMap<String, usize>,
        done: &mut [bool],
        visiting: &mut [bool],
        out: &mut Vec<Value>,
        path: &mut Vec<String>,
    ) -> Result<(), String> {
        if done[i] {
            return Ok(());
        }
        if visiting[i] {
            return Err(format!("计划里有循环依赖：{}", path.join(" → ")));
        }
        visiting[i] = true;
        for d in str_list(specs[i].get("depends_on")) {
            if let Some(&j) = by_key.get(&d) {
                path.push(d);
                visit(j, specs, by_key, done, visiting, out, path)?;
                path.pop();
            }
        }
        visiting[i] = false;
        done[i] = true;
        out.push(specs[i].clone());
        Ok(())
    }
    for i in 0..specs.len() {
        let label = specs[i]
            .get("key")
            .or(specs[i].get("title"))
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        visit(
            i,
            specs,
            &by_key,
            &mut done,
            &mut visiting,
            &mut out,
            &mut vec![label],
        )?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_orders_and_detects_cycles() {
        let specs = vec![
            json!({"key": "b", "title": "B", "depends_on": ["a"]}),
            json!({"key": "a", "title": "A"}),
        ];
        let o = topo(&specs).unwrap();
        assert_eq!(o[0]["key"], "a");
        let cyc = vec![
            json!({"key": "a", "depends_on": ["b"]}),
            json!({"key": "b", "depends_on": ["a"]}),
        ];
        assert!(topo(&cyc).unwrap_err().contains("循环"));
    }

    #[test]
    fn str_list_accepts_many_shapes() {
        assert_eq!(str_list(Some(&json!("a, b"))), vec!["a", "b"]);
        assert_eq!(str_list(Some(&json!("[\"x\"]"))), vec!["x"]);
        assert_eq!(str_list(Some(&json!(["y"]))), vec!["y"]);
    }
}
