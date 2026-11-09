//! The **Conversation** (see `CONTEXT.md`): the live session a frontend
//! drives — the current Agent + Turn pump pair, slash commands, session
//! switching (`/new`, `/fork`, `/handoff`), lifecycle reporting around each
//! run and the `/goal` verify-proposal flow.
//!
//! It knows nothing about where lines come from or go to: output goes
//! through a [`Sink`], and the frontend owns the loop that feeds it input,
//! agent events and background events. The headless REPL (stdin/stdout) and
//! the chat gateway (`gateway/worker.rs`, one Conversation per chat) are
//! its two adapters.
//!
//! 🦀 Rust notes
//! - `Conversation<S: Sink>` is *generic* over its output: each frontend
//!   gets its own monomorphized copy, so calling `self.sink.line(..)` is a
//!   direct call, not a virtual one (`Box<dyn Sink>` would also work).
//! - The Agent moves *into* the spawned task while a turn runs and comes
//!   back as the task's return value (`JoinHandle<Agent>`): no lock is ever
//!   held across a turn, and "busy" is simply `self.job.is_some()`.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEvent, FOLLOW_UP, STEER, SteerQueue};
use crate::background::BackgroundEvent;
use crate::commands::{self as cmds, parse_toggle};
use crate::goal::{GoalCommand, parse_goal_args};
use crate::integrations::{Reporters, State, TurnOutcome};
use crate::models::Attachment;
use crate::pump;

pub const HELP_TEXT: &str = "可用命令：/goal <目标>|clear 目标闭环 · /queue <文本> 任务完成后再发 · /compact 压缩上下文 · \
/model [id] 切换模型 · /new 新会话 · /fork [n] 分叉 · /handoff <说明> 交接 · /plan on|off 计划模式 · \
/code-mode on|off · /daily-mode on|off · /btw <问题> 侧问 · /agents [scan|init] 子代理 · /task 任务板 · @名字 <消息> · /tracing-viewer [id|off] · /help · /quit";
pub const DAILY_HELP_TEXT: &str = " · /memory [edit] 查看记忆 · /learn 复盘写进 skill";

/// Where a Conversation's output goes.
pub trait Sink: Send {
    /// One finished line for the user.
    fn line(&mut self, text: &str);
    /// A streamed piece of a `/btw` answer.
    fn btw_chunk(&mut self, text: &str);
    /// The `/btw` answer is complete.
    fn btw_end(&mut self) {}
    /// A task finished and the agent is idle again.
    fn idle(&mut self) {}
    /// The session changed (`/new`, `/fork`, `/handoff`): reset counters.
    fn session_changed(&mut self, _agent: &Agent) {}
    /// A new Agent value replaced the old one (`/fork` resumes the child
    /// file): frontends re-apply what they add to every agent.
    fn agent_replaced(&mut self, _agent: &mut Agent) {}
    /// Whether `/quit` may end the frontend.
    fn allows_quit(&self) -> bool {
        true
    }
}

/// What a started job does.
pub enum Job {
    Task(String, Vec<Attachment>),
    Compact,
}

pub struct Conversation<S: Sink> {
    pub agent: Option<Agent>,
    pub job: Option<JoinHandle<Agent>>,
    pub cancel: CancellationToken,
    pub steer: Arc<SteerQueue>,
    /// Agent events of the running job; the frontend owns the receiver.
    pub tx: mpsc::UnboundedSender<AgentEvent>,
    pub sink: S,
    /// The task that just ran was a /goal proposal round.
    proposing: bool,
    /// Blocked on the user's answer to a verify proposal.
    pub proposals: Option<Vec<String>>,
    pub quit: bool,
    viewer: crate::trace_viewer::Viewer,
    /// The board while the agent is away running a task.
    sup_handle: Option<Arc<crate::subagents::Supervisor>>,
    /// Herdr / OSC 7501 lifecycle reports.
    pub reporters: Arc<Reporters>,
    outcome: TurnOutcome,
    reported_session: String,
}

/// Report the session identity when it changed since the last report.
fn note_session(last: &mut String, reporters: &Reporters, agent: &Agent) {
    if agent.session_id() != last.as_str() {
        *last = agent.session_id().to_string();
        reporters.report_session(last, agent.session_path());
    }
}

pub fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

impl<S: Sink> Conversation<S> {
    pub fn new(
        agent: Agent,
        sink: S,
        tx: mpsc::UnboundedSender<AgentEvent>,
        reporters: Arc<Reporters>,
    ) -> Self {
        let mut c = Self {
            steer: agent.steer.clone(),
            viewer: crate::trace_viewer::Viewer::new(
                agent.session_dir.clone(),
                Some(agent.workdir.clone()),
            ),
            agent: Some(agent),
            job: None,
            cancel: CancellationToken::new(),
            tx,
            sink,
            proposing: false,
            proposals: None,
            quit: false,
            sup_handle: None,
            reporters,
            outcome: TurnOutcome::default(),
            reported_session: String::new(),
        };
        if let Some(a) = &c.agent {
            note_session(&mut c.reported_session, &c.reporters, a);
        }
        c
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// Feed an agent event of the running job through (outcome tracking).
    pub fn observe(&mut self, ev: &AgentEvent) {
        self.outcome.observe(ev);
    }

    pub fn start(&mut self, job: Job) {
        let Some(mut agent) = self.agent.take() else {
            return;
        };
        self.cancel = CancellationToken::new();
        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        self.sup_handle = agent.supervisor.clone();
        note_session(&mut self.reported_session, &self.reporters, &agent);
        self.outcome = TurnOutcome::default();
        self.reporters.report(State::Working, None);
        self.job = Some(tokio::spawn(async move {
            match job {
                Job::Task(text, atts) => pump::run_task(&mut agent, text, atts, &tx, &cancel).await,
                Job::Compact => {
                    agent.compact("manual", &tx, &cancel).await;
                    agent.checkpoint();
                }
            }
            agent
        }));
    }

    /// Wait for the running job (if any); `None` when idle. Frontends call
    /// it inside `tokio::select!` guarded by [`Conversation::busy`].
    pub async fn join(&mut self) -> Option<Result<Agent, tokio::task::JoinError>> {
        match self.job.as_mut() {
            Some(j) => Some(j.await),
            None => None,
        }
    }

    /// Interrupt the running job (Esc / Ctrl+C / `/stop`).
    pub fn interrupt(&mut self) {
        if self.busy() {
            self.cancel.cancel();
        }
    }

    pub fn job_done(&mut self, agent: Agent) {
        self.sup_handle = agent.supervisor.clone();
        self.agent = Some(agent);
        self.job = None;
        self.sink.idle();
        let o = std::mem::take(&mut self.outcome);
        self.reporters.report(o.state, o.message.as_deref());
        if std::mem::take(&mut self.proposing) {
            self.confirm_proposal();
            if self.proposals.is_some() {
                self.reporters
                    .report(State::Blocked, Some("等待确认 /goal 验收命令"));
            }
        }
        self.maybe_wake_main();
    }

    /// Stop the job and hand the agent back for a final checkpoint.
    pub async fn shutdown(&mut self) {
        if let Some(job) = self.job.take() {
            self.cancel.cancel();
            if let Ok(a) = job.await {
                self.agent = Some(a);
            }
        }
        if let Some(a) = self.agent.as_mut() {
            a.checkpoint();
            a.trace.flush();
        }
        self.viewer.stop();
        if let Some(sup) = self.sup() {
            sup.stop();
        }
    }

    fn confirm_proposal(&mut self) {
        let Some(agent) = &self.agent else { return };
        if pump::live_goal(agent).is_none() {
            return;
        }
        match pump::take_proposal(agent) {
            Some(cmds_) if cmds_.is_empty() => self.sink.line(cmds::NO_VERIFY_GATE),
            found => {
                let list = found.unwrap_or_default();
                if list.is_empty() {
                    self.sink.line("未能解析模型的验收提议。");
                } else {
                    self.sink.line("模型提议的验收命令：");
                    for (i, c) in list.iter().enumerate() {
                        self.sink.line(&format!("  {}. {c}", i + 1));
                    }
                }
                self.sink
                    .line("回复序号采纳；或直接输入一行作为验收命令；/skip 跳过自动验收");
                self.proposals = Some(list);
            }
        }
    }

    fn answer_verify(&mut self, text: &str, proposals: Vec<String>) {
        let t = text.trim();
        if t.is_empty() || t == "/skip" {
            self.sink.line(cmds::VERIFY_SKIPPED);
            self.reporters.report(State::Idle, None);
            return;
        }
        if t.starts_with('/') {
            self.sink
                .line("正在等待验收确认：回复序号、验收命令，或 /skip 跳过");
            self.proposals = Some(proposals);
            return;
        }
        let command = match t.parse::<usize>() {
            Ok(i) if (1..=proposals.len()).contains(&i) => proposals[i - 1].clone(),
            Ok(_) => {
                self.sink.line(&format!(
                    "序号超出范围（1–{}）；或直接输入验收命令，/skip 跳过",
                    proposals.len()
                ));
                self.proposals = Some(proposals);
                return;
            }
            Err(_) => t.to_string(),
        };
        let Some(agent) = self.agent.as_mut() else {
            return;
        };
        if let Some(prompt) = pump::set_verify(agent, &command) {
            self.sink.line(&format!("验收方式已确认：{command}"));
            self.start(Job::Task(prompt, vec![]));
        }
    }

    /// One line of user input (plus attachments from a chat message).
    pub async fn submit(&mut self, line: String, attachments: Vec<Attachment>) {
        if let Some(proposals) = self.proposals.take() {
            return self.answer_verify(&line, proposals);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() && attachments.is_empty() {
            return;
        }
        let (name, arg) = match trimmed.strip_prefix('/') {
            Some(rest) if !line.starts_with(' ') => rest
                .split_once(char::is_whitespace)
                .map_or((rest, ""), |(c, a)| (c, a.trim())),
            _ => ("", ""),
        };
        let name = name.to_lowercase();
        // Board commands and @name never touch the main history: fine mid-turn.
        if matches!(name.as_str(), "agents" | "task") {
            let msg = match self.sup() {
                None => "子代理已关闭（config.toml 的 [subagents] enabled = false）".to_string(),
                Some(sup) if name == "agents" => crate::subagents::agents_command(&sup, arg),
                Some(sup) => crate::subagents::task_command(&sup, arg),
            };
            self.sink.line(&msg);
            return;
        }
        if name.is_empty()
            && let Some(sup) = self.sup()
            && let Some(reply) = crate::subagents::direct_message(&sup, &line)
        {
            self.sink.line(&format!("➤ {reply}"));
            return;
        }
        if self.busy() {
            match name.as_str() {
                "" => {
                    self.steer
                        .push_item(line.clone(), attachments, STEER, "user", vec![]);
                    self.sink
                        .line(&format!("➤ 已排队插话：{}", first_line(&line)));
                }
                "queue" if !arg.is_empty() => {
                    self.steer
                        .push_item(arg.to_string(), attachments, FOLLOW_UP, "user", vec![]);
                    self.sink.line(&format!(
                        "➤ 已排队，当前任务完成后处理：{}",
                        first_line(arg)
                    ));
                }
                "help" => self.sink.line(HELP_TEXT),
                "tracing-viewer" => {
                    let msg = self.viewer.command(arg, Some(&self.reported_session));
                    self.sink.line(&msg);
                }
                _ => self
                    .sink
                    .line("当前任务进行中，该命令暂不可用（/stop 或 Ctrl+C 中断，或等待完成）"),
            }
            return;
        }
        if name.is_empty() {
            return self.start(Job::Task(line, attachments));
        }
        self.command(&name, arg).await;
        if let Some(a) = &self.agent {
            note_session(&mut self.reported_session, &self.reporters, a);
        }
    }

    async fn command(&mut self, name: &str, arg: &str) {
        let Some(agent) = self.agent.as_mut() else {
            return;
        };
        match name {
            "quit" | "exit" if self.sink.allows_quit() => self.quit = true,
            "help" => {
                let daily = agent.daily_mode();
                self.sink.line(&format!(
                    "{HELP_TEXT}{}",
                    if daily { DAILY_HELP_TEXT } else { "" }
                ));
            }
            "queue" => {
                if arg.is_empty() {
                    self.sink.line("用法：/queue <文本>");
                } else {
                    self.start(Job::Task(arg.to_string(), vec![]));
                }
            }
            "new" => {
                agent.new_session();
                self.sink.session_changed(agent);
                self.sink.line("已开始新会话");
            }
            "compact" => self.start(Job::Compact),
            "model" => {
                if arg.is_empty() {
                    let m = agent.client.model_id().to_string();
                    self.sink.line(&format!("当前模型：{m}"));
                } else {
                    let (model, level) = crate::llm::catalog::split_model_arg(arg);
                    match agent.set_model(&model) {
                        Ok(()) => {
                            if level.is_some() {
                                agent.set_thinking(level);
                            }
                            let m = agent.client.model_id().to_string();
                            self.sink.line(&format!("已切换到 {m}"));
                        }
                        Err(e) => self.sink.line(&format!("✗ {e}")),
                    }
                }
            }
            "fork" => {
                if !arg.is_empty() && arg.parse::<usize>().is_err() {
                    return self
                        .sink
                        .line("用法：/fork [n]（n = 保留前 n 轮对话，省略则全部保留）");
                }
                let parent = agent.session_id().to_string();
                let dir = agent.session_dir.clone();
                match agent.fork(arg.parse().ok()) {
                    Ok(path) => match Agent::resume(agent.config.clone(), &path, &dir) {
                        Ok(mut a) => {
                            self.sink.agent_replaced(&mut a);
                            self.steer = a.steer.clone();
                            let id = a.session_id().to_string();
                            self.sink.session_changed(&a);
                            self.agent = Some(a);
                            self.sink
                                .line(&format!("已从会话 {parent} 分叉，新会话 {id}"));
                        }
                        Err(e) => self.sink.line(&format!("✗ {e}")),
                    },
                    Err(e) => self.sink.line(&format!("✗ {e}")),
                }
            }
            "handoff" => {
                if arg.is_empty() {
                    return self.sink.line("用法：/handoff <交接说明>");
                }
                let parent = agent.session_id().to_string();
                let prompt = agent.handoff_prompt(arg);
                let meta = agent.handoff_meta();
                agent.reset_to(meta);
                self.sink.session_changed(agent);
                let id = agent.session_id().to_string();
                self.sink
                    .line(&format!("已交接：新会话 {id}，关联原会话 {parent}"));
                self.start(Job::Task(prompt, vec![]));
            }
            "plan" => match parse_toggle(arg, agent.plan_mode()) {
                None => self.sink.line(cmds::PLAN_USAGE),
                Some(on) if agent.set_plan_mode(on) => self.sink.line(cmds::plan_notice(on)),
                Some(on) => self.sink.line(&cmds::unchanged_notice("计划模式", on)),
            },
            "code-mode" => match parse_toggle(arg, agent.code_mode()) {
                None => self.sink.line(cmds::CODE_MODE_USAGE),
                Some(on) if agent.set_code_mode(on) => self.sink.line(cmds::code_mode_notice(on)),
                Some(on) => self.sink.line(&cmds::unchanged_notice("Code mode ", on)),
            },
            "daily-mode" => match parse_toggle(arg, agent.daily_mode()) {
                None => self.sink.line(cmds::DAILY_USAGE),
                Some(on) if agent.set_daily_mode(on) => {
                    self.sink.line(cmds::daily_toggle_notice(on));
                    if let Some(n) = cmds::web_search_missing_notice(agent) {
                        self.sink.line(&n);
                    }
                }
                Some(on) => self.sink.line(&cmds::unchanged_notice("Daily mode ", on)),
            },
            "memory" | "learn" if !agent.daily_mode() => self
                .sink
                .line(&cmds::daily_only_notice(&format!("/{name}"))),
            "memory" => match crate::memory::parse_memory_command(arg) {
                Err(()) => self.sink.line(crate::memory::MEMORY_USAGE),
                Ok(target) => {
                    let store = agent.memory_store();
                    match (target, &store) {
                        (Some(t), Some(s)) => self.sink.line(&format!(
                            "直接编辑 {}，从下一个会话起生效",
                            s.path(&t).display()
                        )),
                        _ => {
                            let text =
                                crate::memory::describe(store.as_ref(), agent.memory.as_ref());
                            self.sink.line(&text);
                        }
                    }
                }
            },
            "learn" => {
                let msg = crate::learn::start_review(agent, false);
                self.sink.line(&msg);
            }
            "btw" => {
                if arg.is_empty() {
                    return self.sink.line("用法：/btw <问题>");
                }
                let mut rx = cmds::btw_query(&agent.client, &agent.messages, arg);
                self.sink.line(&format!("💬 btw · {arg}"));
                while let Some(item) = rx.recv().await {
                    match item {
                        Ok(t) => self.sink.btw_chunk(&t),
                        Err(e) => {
                            self.sink.btw_end();
                            self.sink.line(&format!("✗ {e}"));
                        }
                    }
                }
                self.sink.btw_end();
                self.sink.line("（侧问答案不会进入会话）");
            }
            "goal" => match parse_goal_args(arg) {
                GoalCommand::Query => match pump::live_goal(agent) {
                    None => self.sink.line("当前没有 goal"),
                    Some(g) => {
                        let t = cmds::goal_status_text(g);
                        self.sink.line(&t);
                    }
                },
                GoalCommand::Clear => {
                    if pump::clear_goal(agent) {
                        self.sink.line("已退出 goal 模式");
                    } else {
                        self.sink.line("当前没有 goal");
                    }
                }
                GoalCommand::Set(text) => {
                    let prompt = pump::set_goal(agent, &text);
                    self.proposing = true;
                    self.start(Job::Task(prompt, vec![]));
                }
            },
            "tracing-viewer" => {
                let id = agent.session_id().to_string();
                let msg = self.viewer.command(arg, Some(&id));
                self.sink.line(&msg);
            }
            other => self
                .sink
                .line(&format!("未知命令 /{other}（/help 查看可用命令）")),
        }
    }

    /// The user-facing lines for a background event (empty: nothing to show).
    pub fn background_lines(&self, ev: &BackgroundEvent) -> Vec<String> {
        match ev {
            BackgroundEvent::Notice { kind, text } if !text.is_empty() => {
                let icon = match kind.as_str() {
                    "result" => "✓",
                    "alert" => "⚠",
                    "question" => "?",
                    _ => "◆",
                };
                vec![format!("{icon} {text}")]
            }
            BackgroundEvent::PlanProposed(_) => {
                let Some(sup) = self.sup() else {
                    return vec![];
                };
                let mut out = vec!["◆ 子任务计划待确认：".to_string()];
                out.extend(sup.status_text(None).lines().map(|l| format!("  {l}")));
                out.push("  /task approve all 开始 · /task reject all 放弃".into());
                out
            }
            _ => vec![],
        }
    }

    /// After a background event was shown: wake an idle main agent.
    pub fn after_background(&mut self, ev: &BackgroundEvent) {
        if matches!(ev, BackgroundEvent::MainWake) {
            self.maybe_wake_main();
        }
    }

    pub fn sup(&self) -> Option<Arc<crate::subagents::Supervisor>> {
        self.agent
            .as_ref()
            .and_then(|a| a.supervisor.clone())
            .or_else(|| self.sup_handle.clone())
    }

    /// Start a main-agent turn for subagent notices queued while idle.
    pub fn maybe_wake_main(&mut self) {
        if self.busy() || self.proposals.is_some() || self.quit {
            return;
        }
        let Some(text) = self.sup().and_then(|s| s.take_wake_input()) else {
            return;
        };
        self.sink.line("◆ 子代理有新进展，交给主 agent 处理");
        self.start(Job::Task(text, vec![]));
    }
}
