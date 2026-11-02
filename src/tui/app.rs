//! TUI application state: keys, slash commands, agent events.

use crate::integrations::{Reporters, State, TurnOutcome};
use std::sync::Arc;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::chat::{Entry, EntryKind, NoticeKind, ToolStatus, entries_from_history};
use super::clipboard;
use super::fx::Fx;
use super::game2048::{Dir, Game};
use super::input::Input;
use super::paste::{self, PasteStore};
use super::theme::{self, THEMES, Theme};
use crate::agent::{Agent, AgentEvent, EventTx, FOLLOW_UP, STEER, SteerQueue};
use crate::background::BackgroundEvent;
use crate::commands::{self as cmds, parse_toggle};
use crate::config::{self, Config};
use crate::goal::{GoalCommand, parse_goal_args};
use crate::llm::catalog;
use crate::models::Attachment;
use crate::pump;
use crate::session::{self, SessionMeta};
use crate::skills::{Skill, discover_skills};
use crate::subagents::Supervisor;

/// What the agent is doing right now (drives the status bar).
#[derive(Debug, Clone, PartialEq)]
pub enum Activity {
    Idle,
    Thinking,
    Writing,
    Tool(String),
    Compacting,
}

/// Status-bar facts mirrored from the agent (which is away while a turn runs).
#[derive(Debug, Clone, Default)]
pub struct Info {
    pub model: String,
    pub session_id: String,
    pub title: String,
    pub context_tokens: u64,
    pub context_window: u64,
    pub session_total: u64,
    pub plan_mode: bool,
    pub daily_mode: bool,
    pub code_mode: bool,
    /// `🎯 2/8` while a goal is live.
    pub goal: Option<String>,
    pub thinking: Option<String>,
    pub workdir: String,
}

pub enum Popup {
    Help,
    Sessions {
        items: Vec<SessionMeta>,
        selected: usize,
        all: bool,
    },
    Models {
        items: Vec<(String, String)>,
        selected: usize,
        filter: String,
    },
    /// After a /goal proposal round: pick the verify command.
    VerifyPicker {
        items: Vec<String>,
        selected: usize,
    },
    /// A /btw side answer (never enters the session).
    Btw {
        question: String,
        answer: String,
        done: bool,
        scroll: u16,
    },
    /// Read-only text (/task, /agents scan, /memory …).
    Text {
        title: String,
        body: String,
        scroll: u16,
    },
    Game(Box<Game>),
    /// Paste-a-key dialog for a provider with no key (from the model
    /// picker); the key is checked before it is saved to auth.json.
    Auth {
        provider: String,
        model: String,
        input: String,
        status: Option<String>,
        checking: bool,
    },
    /// The subagent board: tasks on the left, the selected one's transcript.
    Agents {
        selected: usize,
        scroll: u16,
    },
}

pub const VERIFY_EDIT: &str = "✏️ 自己输入/编辑命令…";
pub const VERIFY_SKIP: &str = "⏭️ 跳过（不自动验收）";

pub enum Job {
    Turn(String, Vec<Attachment>),
    Compact,
}

pub struct SlashCommand {
    pub name: &'static str,
    pub args: &'static str,
    pub help: &'static str,
}

pub const COMMANDS: &[SlashCommand] = &[
    SlashCommand {
        name: "help",
        args: "",
        help: "快捷键与命令",
    },
    SlashCommand {
        name: "new",
        args: "",
        help: "开始新会话",
    },
    SlashCommand {
        name: "sessions",
        args: "",
        help: "浏览并恢复历史会话",
    },
    SlashCommand {
        name: "fork",
        args: "[n]",
        help: "把当前会话（或前 n 轮）复制成新会话",
    },
    SlashCommand {
        name: "handoff",
        args: "<note>",
        help: "带着一段交接说明开一个干净的新会话",
    },
    SlashCommand {
        name: "compact",
        args: "",
        help: "立即压缩上下文",
    },
    SlashCommand {
        name: "model",
        args: "[id]",
        help: "切换模型",
    },
    SlashCommand {
        name: "thinking",
        args: "[level]",
        help: "设置思考强度（无参数则轮换）",
    },
    SlashCommand {
        name: "plan",
        args: "[on|off]",
        help: "只读计划模式",
    },
    SlashCommand {
        name: "goal",
        args: "<目标>|clear",
        help: "目标闭环：验收命令不过就继续改",
    },
    SlashCommand {
        name: "queue",
        args: "<文本>",
        help: "排队到当前任务全部完成后（同 alt+enter）",
    },
    SlashCommand {
        name: "code-mode",
        args: "[on|off]",
        help: "让模型写 Python 批量调用工具",
    },
    SlashCommand {
        name: "daily-mode",
        args: "[on|off]",
        help: "日常助理模式：网页、记忆、历史会话",
    },
    SlashCommand {
        name: "memory",
        args: "[edit [user|notes]]",
        help: "查看/编辑长期记忆（daily）",
    },
    SlashCommand {
        name: "learn",
        args: "",
        help: "后台复盘，把经验写进 skill（daily）",
    },
    SlashCommand {
        name: "btw",
        args: "<问题>",
        help: "侧问：不打断任务、不进历史",
    },
    SlashCommand {
        name: "agents",
        args: "[scan|init]",
        help: "子代理面板 · 项目画像 · 生成项目角色",
    },
    SlashCommand {
        name: "task",
        args: "[id|动作 id]",
        help: "任务板：查看、批准、暂停、取消、重试…",
    },
    SlashCommand {
        name: "tracing-viewer",
        args: "[id|off]",
        help: "在浏览器里看本会话的 trace",
    },
    SlashCommand {
        name: "copy",
        args: "",
        help: "复制最后一条回复到剪贴板",
    },
    SlashCommand {
        name: "2048",
        args: "",
        help: "等待时玩一局 2048",
    },
    SlashCommand {
        name: "export",
        args: "[path]",
        help: "导出对话为 Markdown",
    },
    SlashCommand {
        name: "theme",
        args: "[neon|aurora|ember|dawn|omarchy]",
        help: "切换主题",
    },
    SlashCommand {
        name: "mouse",
        args: "",
        help: "切换鼠标捕获（关闭后可用终端原生选择复制）",
    },
    SlashCommand {
        name: "clear",
        args: "",
        help: "清空屏幕（不影响会话历史）",
    },
    SlashCommand {
        name: "quit",
        args: "",
        help: "退出",
    },
];

pub struct App {
    pub theme: Theme,
    pub config: Config,
    agent: Option<Agent>,
    pub job: Option<JoinHandle<Agent>>,
    cancel: CancellationToken,
    steer: Arc<SteerQueue>,
    tx: EventTx,
    pub rx: mpsc::UnboundedReceiver<AgentEvent>,
    pub info: Info,
    pub entries: Vec<Entry>,
    pub input: Input,
    /// Rows scrolled up from the bottom (0 = follow the tail).
    pub scroll: usize,
    pub max_scroll: usize,
    pub expand_tools: bool,
    pub show_thinking: bool,
    pub busy_since: Option<Instant>,
    pub activity: Activity,
    pub popup: Option<Popup>,
    pub cmd_selected: usize,
    pub frame: usize,
    /// Where the prompt caret was drawn last frame (see `ui::draw_input`).
    pub cursor: Option<ratatui::layout::Position>,
    pub fx: Fx,
    pub quit: bool,
    last_ctrl_c: Option<Instant>,
    pub mouse_capture: bool,
    pub mouse_changed: bool,
    pub toast: Option<(String, Instant)>,
    pub skills: Vec<Skill>,
    pub dirty: bool,
    /// Collapsed pastes and attachment markers of the prompt box.
    pub paste: PasteStore,
    /// The task that just ran was a /goal proposal round.
    proposing: bool,
    /// The next submission is the (edited) verify command.
    pub editing_verify: bool,
    /// `[queue] after_failed_run` held the queue: do not auto-continue.
    queue_held: bool,
    pub bg: Option<mpsc::UnboundedReceiver<BackgroundEvent>>,
    /// The task board (kept while the agent is away running a turn).
    pub sup: Option<Arc<Supervisor>>,
    pub btw_rx: Option<mpsc::UnboundedReceiver<Result<String, String>>>,
    /// `/memory edit`: the event loop suspends the TUI and opens an editor.
    pub pending_editor: Option<std::path::PathBuf>,
    /// OSC 52 copy request for the event loop to write.
    pub pending_osc52: Option<String>,
    /// The user's theme (daily mode may swap in Dawn on top of it).
    base_theme: Theme,
    /// History at the start of the running task (for /btw while busy).
    busy_snapshot: Option<(crate::llm::LlmClient, Vec<crate::models::Message>)>,
    viewer: crate::trace_viewer::Viewer,
    /// Herdr / OSC 7501 lifecycle reports.
    reporters: std::sync::Arc<Reporters>,
    outcome: TurnOutcome,
    reported_session: String,
}

impl App {
    pub fn new(agent: Agent, config: Config) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let theme = theme::by_name(config.ui.theme.as_deref());
        let motion = config
            .ui
            .motion
            .clone()
            .or_else(|| std::env::var("LIMBO_MOTION").ok());
        let entries = entries_from_history(&agent.messages);
        let skills = discover_skills(&agent.workdir, agent.daily_mode());
        let sup = agent.supervisor.clone();
        let viewer = crate::trace_viewer::Viewer::new(
            agent.session_dir.clone(),
            Some(agent.workdir.clone()),
        );
        let mut app = Self {
            base_theme: theme,
            busy_snapshot: None,
            paste: PasteStore::default(),
            proposing: false,
            editing_verify: false,
            queue_held: false,
            bg: crate::background::take_receiver(),
            sup,
            btw_rx: None,
            pending_editor: None,
            pending_osc52: None,
            viewer,
            reporters: {
                let r = std::sync::Arc::new(Reporters::from_env(true));
                crate::integrations::install_exit_hooks(r.clone());
                r
            },
            outcome: TurnOutcome::default(),
            reported_session: String::new(),
            theme,
            fx: Fx::new(motion.as_deref()),
            config,
            steer: agent.steer.clone(),
            agent: Some(agent),
            job: None,
            cancel: CancellationToken::new(),
            tx,
            rx,
            info: Info::default(),
            entries,
            input: Input::default(),
            scroll: 0,
            max_scroll: 0,
            expand_tools: false,
            show_thinking: false,
            busy_since: None,
            activity: Activity::Idle,
            popup: None,
            cmd_selected: 0,
            frame: 0,
            cursor: None,
            quit: false,
            last_ctrl_c: None,
            mouse_capture: true,
            mouse_changed: false,
            toast: None,
            skills,
            dirty: true,
        };
        app.refresh_info();
        app.apply_look();
        if let Some(sup) = &app.sup
            && !sup.interrupted_on_load.is_empty()
            && app.config.subagents.on_resume != "auto"
        {
            let ids = sup.interrupted_on_load.join("、");
            app.notice(
                NoticeKind::Warn,
                format!(
                    "{} 个子任务在上次退出时中断（{ids}），已暂停；/task resume all 继续",
                    sup.interrupted_on_load.len()
                ),
            );
        }
        if !app.steer.is_empty() {
            app.notice(
                NoticeKind::Info,
                format!(
                    "恢复了 {} 条排队消息，发送任意内容即可继续",
                    app.steer.len()
                ),
            );
        }
        app
    }

    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    pub fn queued(&self) -> Vec<String> {
        self.steer
            .items()
            .into_iter()
            .filter(|i| i.source == "user")
            .map(|i| {
                if i.mode == FOLLOW_UP {
                    format!("⏭ {}", i.text)
                } else {
                    i.text
                }
            })
            .collect()
    }

    /// `2 运行中 · 1 待确认` for the status bar.
    pub fn tasks_label(&self) -> String {
        self.sup
            .as_ref()
            .map(|s| s.counts_label())
            .unwrap_or_default()
    }

    fn refresh_info(&mut self) {
        if let Some(a) = &self.agent
            && a.session_id() != self.reported_session
        {
            self.reported_session = a.session_id().to_string();
            self.reporters
                .report_session(&self.reported_session, a.session_path());
        }
        if let Some(a) = &self.agent {
            self.info = Info {
                model: a.client.model_id().to_string(),
                session_id: a.session_id().to_string(),
                title: a.meta.title.clone(),
                context_tokens: a.context_tokens(),
                context_window: a.context_window(),
                session_total: a.session_total_tokens(),
                plan_mode: a.plan_mode(),
                daily_mode: a.daily_mode(),
                code_mode: a.code_mode(),
                goal: pump::live_goal(a).map(|g| {
                    if g.verify_command.is_empty() {
                        "🎯 goal".to_string()
                    } else {
                        format!("🎯 {}/{}", g.rounds_completed, g.max_rounds)
                    }
                }),
                thinking: a.client.thinking_effort.clone(),
                workdir: a.workdir.display().to_string(),
            };
        }
    }

    /// Daily mode's look: Dawn swaps the palette, "badge" only shows a pill.
    fn apply_look(&mut self) {
        let dawn = self.info.daily_mode && self.config.daily.look != "badge";
        let next = if dawn { theme::DAWN } else { self.base_theme };
        if next.name != self.theme.name {
            self.theme = next;
            for e in &mut self.entries {
                e.touch();
            }
            self.fx.theme_changed(&self.theme);
        }
    }

    pub fn notice(&mut self, kind: NoticeKind, text: impl Into<String>) {
        self.entries.push(Entry::new(EntryKind::Notice {
            kind,
            text: text.into(),
        }));
        self.scroll = 0;
        self.dirty = true;
    }

    pub fn toast(&mut self, text: impl Into<String>) {
        self.toast = Some((text.into(), Instant::now()));
        self.dirty = true;
    }

    pub fn animating(&self) -> bool {
        self.busy() || self.fx.running() || self.toast.is_some() || self.entries.is_empty()
    }

    // -- jobs ----------------------------------------------------------------

    fn start_job(&mut self, job: Job) {
        let Some(mut agent) = self.agent.take() else {
            return;
        };
        self.cancel = CancellationToken::new();
        let cancel = self.cancel.clone();
        let tx = self.tx.clone();
        self.busy_since = Some(Instant::now());
        self.sup = agent.supervisor.clone();
        self.busy_snapshot = Some((agent.client.clone(), agent.messages.clone()));
        self.note_session_of(&agent);
        self.outcome = TurnOutcome::default();
        self.reporters.report(State::Working, None);
        self.activity = if matches!(job, Job::Compact) {
            Activity::Compacting
        } else {
            Activity::Thinking
        };
        self.scroll = 0;
        // The agent moves into the task and comes back when it finishes.
        self.job = Some(tokio::spawn(async move {
            match job {
                Job::Turn(text, atts) => pump::run_task(&mut agent, text, atts, &tx, &cancel).await,
                Job::Compact => {
                    agent.compact("manual", &tx, &cancel).await;
                    agent.checkpoint();
                }
            }
            agent
        }));
        self.dirty = true;
    }

    /// Called when the running job hands the agent back.
    pub fn job_done(&mut self, agent: Agent) {
        self.agent = Some(agent);
        self.job = None;
        self.busy_since = None;
        self.activity = Activity::Idle;
        for e in &mut self.entries {
            match &mut e.kind {
                EntryKind::Assistant { streaming, .. } if *streaming => {
                    *streaming = false;
                    e.touch();
                }
                EntryKind::Thinking {
                    live,
                    started,
                    secs,
                    ..
                } if *live => {
                    *live = false;
                    *secs = Some(started.elapsed().as_secs_f64());
                    e.touch();
                }
                EntryKind::Tool { status, .. } if *status == ToolStatus::Running => {
                    *status = ToolStatus::Failed;
                    e.touch();
                }
                _ => {}
            }
        }
        self.refresh_info();
        self.fx.turn_done(&self.theme);
        let o = std::mem::take(&mut self.outcome);
        self.reporters.report(o.state, o.message.as_deref());
        if std::mem::take(&mut self.proposing) {
            self.confirm_proposal();
        }
        // Something was queued after the task's last drain: keep going.
        if !self.queue_held
            && self.popup.is_none()
            && self.steer.items().iter().any(|i| i.source == "user")
        {
            self.start_job(Job::Turn(String::new(), vec![]));
        } else {
            self.maybe_wake_main();
        }
        self.dirty = true;
    }

    /// After a proposal round: show the verify picker (or say why not).
    fn confirm_proposal(&mut self) {
        let Some(agent) = &self.agent else { return };
        if pump::live_goal(agent).is_none() {
            return;
        }
        match pump::take_proposal(agent) {
            Some(list) if list.is_empty() => self.notice(NoticeKind::Info, cmds::NO_VERIFY_GATE),
            found => {
                let mut items = found.unwrap_or_default();
                if items.is_empty() {
                    self.notice(
                        NoticeKind::Warn,
                        "未能解析模型的验收提议，可以自己输入验收命令",
                    );
                }
                items.push(VERIFY_EDIT.into());
                items.push(VERIFY_SKIP.into());
                self.popup = Some(Popup::VerifyPicker { items, selected: 0 });
                self.fx.popup_opened();
                self.reporters
                    .report(State::Blocked, Some("等待确认 /goal 验收命令"));
            }
        }
    }

    fn on_auth_checked(
        &mut self,
        provider: &str,
        model: &str,
        key: &str,
        detail: String,
        error: bool,
    ) {
        if let Some(Popup::Auth {
            status, checking, ..
        }) = &mut self.popup
        {
            if error {
                // A definite rejection keeps the dialog open.
                *checking = false;
                *status = Some(format!("✗ {detail}"));
                return;
            }
        } else {
            return; // dialog closed meanwhile
        }
        self.popup = None;
        if let Err(e) = crate::auth::save(provider, key, None) {
            self.notice(NoticeKind::Error, format!("保存 key 失败：{e}"));
            return;
        }
        self.notice(
            NoticeKind::Info,
            format!(
                "已保存 {} 的 key {}（{detail}）",
                catalog::provider_display_name(provider),
                crate::auth::mask(key)
            ),
        );
        self.set_model(model);
    }

    /// Report the session identity when it changed since the last report.
    fn note_session_of(&mut self, agent: &Agent) {
        if agent.session_id() != self.reported_session {
            self.reported_session = agent.session_id().to_string();
            self.reporters
                .report_session(&self.reported_session, agent.session_path());
        }
    }

    fn skip_verify(&mut self) {
        self.notice(NoticeKind::Info, cmds::VERIFY_SKIPPED);
        self.reporters.report(State::Idle, None);
    }

    fn accept_verify(&mut self, command: &str) {
        let Some(agent) = self.agent.as_mut() else {
            return;
        };
        if let Some(prompt) = pump::set_verify(agent, command) {
            self.notice(NoticeKind::Info, format!("验收方式已确认：{command}"));
            self.refresh_info();
            self.start_job(Job::Turn(prompt, vec![]));
        }
    }

    /// Start a main-agent turn for subagent notices queued while idle.
    fn maybe_wake_main(&mut self) {
        if self.busy() || self.popup.is_some() || self.editing_verify {
            return;
        }
        let Some(text) = self.sup.as_ref().and_then(|s| s.take_wake_input()) else {
            return;
        };
        self.notice(NoticeKind::Info, "◆ 子代理有新进展，交给主 agent 处理");
        self.start_job(Job::Turn(text, vec![]));
    }

    pub fn on_background(&mut self, ev: BackgroundEvent) {
        self.dirty = true;
        if let BackgroundEvent::AuthChecked {
            provider,
            model,
            key,
            detail,
            error,
        } = ev
        {
            return self.on_auth_checked(&provider, &model, &key, detail, error);
        }
        match ev {
            BackgroundEvent::Notice { kind, text } if !text.is_empty() => {
                let (k, icon) = match kind.as_str() {
                    "result" => (NoticeKind::Info, "✓"),
                    "alert" => (NoticeKind::Warn, "⚠"),
                    "question" => (NoticeKind::Warn, "?"),
                    _ => (NoticeKind::Info, "◆"),
                };
                self.notice(k, format!("{icon} {text}"));
            }
            BackgroundEvent::PlanProposed(ids) => {
                self.notice(
                    NoticeKind::Info,
                    format!("◆ 子任务计划待确认（{}）：/task 查看 · /task approve all 开始 · /task reject all 放弃", ids.join(", ")),
                );
            }
            BackgroundEvent::MainWake => self.maybe_wake_main(),
            BackgroundEvent::AuthChecked { .. } => {}
            _ => {}
        }
    }

    pub fn on_btw(&mut self, item: Option<Result<String, String>>) {
        self.dirty = true;
        let Some(Popup::Btw { answer, done, .. }) = &mut self.popup else {
            self.btw_rx = None;
            return;
        };
        match item {
            Some(Ok(t)) => answer.push_str(&t),
            Some(Err(e)) => {
                answer.push_str(&format!("\n\n✗ {e}"));
                *done = true;
                self.btw_rx = None;
            }
            None => {
                *done = true;
                self.btw_rx = None;
            }
        }
    }

    // -- agent events ----------------------------------------------------------

    fn close_live(&mut self) {
        if let Some(e) = self.entries.last_mut() {
            match &mut e.kind {
                EntryKind::Thinking {
                    live,
                    started,
                    secs,
                    ..
                } if *live => {
                    *live = false;
                    *secs = Some(started.elapsed().as_secs_f64());
                    e.touch();
                }
                EntryKind::Assistant { streaming, .. } if *streaming => {
                    *streaming = false;
                    e.touch();
                }
                _ => {}
            }
        }
    }

    pub fn on_agent_event(&mut self, ev: AgentEvent) {
        self.dirty = true;
        self.outcome.observe(&ev);
        match ev {
            AgentEvent::TurnStart => self.activity = Activity::Thinking,
            AgentEvent::Text(t) => {
                self.activity = Activity::Writing;
                if let Some(Entry {
                    kind:
                        EntryKind::Assistant {
                            text,
                            streaming: true,
                        },
                    ..
                }) = self.entries.last_mut()
                {
                    text.push_str(&t);
                    self.entries.last_mut().unwrap().touch();
                } else {
                    self.close_live();
                    self.entries.push(Entry::new(EntryKind::Assistant {
                        text: t,
                        streaming: true,
                    }));
                }
            }
            AgentEvent::Thinking(t) => {
                self.activity = Activity::Thinking;
                if let Some(Entry {
                    kind:
                        EntryKind::Thinking {
                            text, live: true, ..
                        },
                    ..
                }) = self.entries.last_mut()
                {
                    text.push_str(&t);
                    self.entries.last_mut().unwrap().touch();
                } else {
                    self.close_live();
                    self.entries.push(Entry::new(EntryKind::Thinking {
                        text: t,
                        live: true,
                        started: Instant::now(),
                        secs: None,
                    }));
                }
            }
            AgentEvent::Retry { attempt, reason } => {
                // Drop the partial answer that is about to be re-sent.
                if matches!(
                    self.entries.last().map(|e| &e.kind),
                    Some(EntryKind::Assistant {
                        streaming: true,
                        ..
                    })
                ) {
                    self.entries.pop();
                }
                self.notice(
                    NoticeKind::Warn,
                    format!(
                        "连接中断，正在重试（第 {attempt} 次）：{}",
                        crate::llm::truncate_chars(&reason, 160)
                    ),
                );
            }
            AgentEvent::ToolCall {
                id,
                name,
                arguments,
            } => {
                self.close_live();
                self.activity = Activity::Tool(name.clone());
                self.entries.push(Entry::new(EntryKind::Tool {
                    id,
                    name,
                    args: arguments,
                    status: ToolStatus::Running,
                    output: String::new(),
                    progress: vec![],
                    started: Instant::now(),
                    secs: None,
                }));
            }
            AgentEvent::ToolProgress { id, line } => {
                if let Some(e) = self.find_tool(&id)
                    && let EntryKind::Tool { progress, .. } = &mut e.kind
                {
                    progress.push(line);
                    e.touch();
                }
            }
            AgentEvent::ToolResult {
                id,
                success,
                output,
                ..
            } => {
                if let Some(e) = self.find_tool(&id)
                    && let EntryKind::Tool {
                        status,
                        output: out,
                        started,
                        secs,
                        ..
                    } = &mut e.kind
                {
                    *status = if success {
                        ToolStatus::Ok
                    } else {
                        ToolStatus::Failed
                    };
                    *out = output;
                    *secs = Some(started.elapsed().as_secs_f64());
                    e.touch();
                }
                self.activity = Activity::Thinking;
            }
            AgentEvent::Compaction {
                compacted,
                trigger,
                reason,
                tokens_before,
                tokens_after,
            } => {
                if compacted {
                    self.notice(
                        NoticeKind::Compaction,
                        format!(
                            "上下文已压缩（{trigger}） {} → {}",
                            fmt_tokens(tokens_before),
                            fmt_tokens(tokens_after)
                        ),
                    );
                    self.fx.compacted(&self.theme);
                } else {
                    self.notice(
                        NoticeKind::Warn,
                        reason.unwrap_or_else(|| "压缩未执行".into()),
                    );
                }
            }
            AgentEvent::Usage {
                context_tokens,
                context_window,
                session_total,
            } => {
                self.info.context_tokens = context_tokens;
                self.info.context_window = context_window;
                self.info.session_total = session_total;
            }
            AgentEvent::Steer(t) => {
                self.close_live();
                self.entries.push(Entry::new(EntryKind::User { text: t }));
            }
            AgentEvent::Info(t) => self.notice(NoticeKind::Info, t),
            AgentEvent::Warning(t) => self.notice(NoticeKind::Warn, t),
            AgentEvent::Error(t) => {
                self.notice(NoticeKind::Error, t);
                self.fx.error(&self.theme);
            }
            AgentEvent::Interrupted => self.notice(NoticeKind::Warn, "已打断"),
            AgentEvent::TurnEnd => self.close_live(),
            ref goal_ev => {
                let kind = match goal_ev {
                    AgentEvent::GoalVerifyResult { exit_code, .. } if *exit_code != Some(0) => {
                        NoticeKind::Warn
                    }
                    AgentEvent::GoalExhausted { .. } | AgentEvent::QueueHeld { .. } => {
                        NoticeKind::Warn
                    }
                    _ => NoticeKind::Info,
                };
                if let AgentEvent::QueueHeld { .. } = goal_ev {
                    self.queue_held = true;
                }
                if let AgentEvent::GoalVerifyStarted { .. } = goal_ev {
                    self.activity = Activity::Tool("verify".into());
                }
                // Mirror the goal badge while the agent is away.
                match goal_ev {
                    AgentEvent::GoalVerifyResult {
                        round, max_rounds, ..
                    } => {
                        self.info.goal = Some(format!("🎯 {round}/{max_rounds}"));
                    }
                    AgentEvent::GoalPassed { .. } => self.info.goal = Some("🎯 ✓".into()),
                    _ => {}
                }
                if let Some(line) = cmds::goal_event_line(goal_ev) {
                    self.notice(kind, line);
                }
                if let AgentEvent::GoalPassed { .. } = goal_ev {
                    self.fx.turn_done(&self.theme);
                }
            }
        }
    }

    fn find_tool(&mut self, id: &str) -> Option<&mut Entry> {
        self.entries
            .iter_mut()
            .rev()
            .find(|e| matches!(&e.kind, EntryKind::Tool { id: i, .. } if i == id))
    }

    // -- input ----------------------------------------------------------------

    /// Re-scan the skill directories (skills can be added while running).
    pub fn refresh_skills(&mut self) {
        if let Some(a) = &self.agent {
            self.skills = discover_skills(&a.workdir, a.daily_mode());
        }
    }

    /// Slash-command suggestions for the current input.
    pub fn command_matches(&self) -> Vec<(String, String, String)> {
        let text = &self.input.text;
        if !text.starts_with('/') || text.contains(' ') || text.contains('\n') {
            return vec![];
        }
        let q = &text[1..];
        let mut out: Vec<(String, String, String)> = COMMANDS
            .iter()
            .filter(|c| c.name.starts_with(q))
            .map(|c| (c.name.to_string(), c.args.to_string(), c.help.to_string()))
            .collect();
        for s in &self.skills {
            if s.name.starts_with(q) && !COMMANDS.iter().any(|c| c.name == s.name) {
                out.push((
                    s.name.clone(),
                    "[args]".into(),
                    format!("skill · {}", s.description),
                ));
            }
        }
        out
    }

    pub fn on_paste(&mut self, text: String) {
        self.dirty = true;
        if let Some(Popup::Models { filter, .. }) = &mut self.popup {
            filter.push_str(text.trim());
            return;
        }
        if self.popup.is_some() {
            return;
        }
        let text = paste::clean(&text);
        // Dragged-in files arrive as a paste of their paths.
        if let Some(paths) = paste::dropped_paths(&text) {
            for p in paths {
                let marker = self.paste.add_file(&p);
                self.input.insert_str(&format!("{marker} "));
            }
            return;
        }
        let shown = self.paste.collapse(&text);
        self.input.insert_str(&shown);
    }

    /// Ctrl+V: images and files from the OS clipboard become attachments.
    fn paste_clipboard(&mut self) {
        match clipboard::read() {
            clipboard::Clip::Image { data, ext } => match clipboard::save_image(&data, &ext) {
                Ok(path) => {
                    let marker = self.paste.add_image(&path, &ext);
                    self.input.insert_str(&format!("{marker} "));
                    if !self.agent.as_ref().is_none_or(|a| a.client.spec.vision) {
                        self.toast("当前模型不支持图片，发送时会附上文件路径");
                    }
                }
                Err(e) => self.toast(format!("保存剪贴板图片失败：{e}")),
            },
            clipboard::Clip::Files(paths) => {
                for p in paths {
                    let marker = self.paste.add_file(&p);
                    self.input.insert_str(&format!("{marker} "));
                }
            }
            clipboard::Clip::Text(t) => self.on_paste(t),
            clipboard::Clip::Empty => self.toast("剪贴板是空的（或没有 wl-paste/xclip/pbpaste）"),
        }
    }

    /// Copy the last assistant reply (native tool, else OSC 52).
    fn copy_last_reply(&mut self) {
        let text = self.entries.iter().rev().find_map(|e| match &e.kind {
            EntryKind::Assistant { text, .. } if !text.trim().is_empty() => Some(text.clone()),
            _ => None,
        });
        let Some(text) = text else {
            self.toast("还没有可复制的回复");
            return;
        };
        if clipboard::write_text(&text) {
            self.toast("已复制最后一条回复");
        } else {
            self.pending_osc52 = Some(clipboard::osc52(&text));
            self.toast("已通过终端（OSC 52）复制");
        }
    }

    pub fn on_mouse(&mut self, m: MouseEvent) {
        match m.kind {
            MouseEventKind::ScrollUp => self.scroll_by(3),
            MouseEventKind::ScrollDown => self.scroll_by(-3),
            _ => return,
        }
        self.dirty = true;
    }

    fn scroll_by(&mut self, delta: isize) {
        self.scroll = (self.scroll as isize + delta).clamp(0, self.max_scroll as isize) as usize;
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.kind == KeyEventKind::Release {
            return;
        }
        self.dirty = true;
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);

        if self.popup.is_some() {
            self.popup_key(key);
            return;
        }

        match key.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy() {
                    self.cancel.cancel();
                    self.toast("正在打断…");
                } else if !self.input.is_empty() {
                    self.input.take();
                } else if self
                    .last_ctrl_c
                    .is_some_and(|t| t.elapsed().as_millis() < 1500)
                {
                    self.quit = true;
                } else {
                    self.last_ctrl_c = Some(Instant::now());
                    self.toast("再按一次 Ctrl+C 退出");
                }
            }
            KeyCode::Char('d') if ctrl && self.input.is_empty() => self.quit = true,
            KeyCode::Esc if self.editing_verify => {
                self.editing_verify = false;
                self.input.take();
                self.skip_verify();
            }
            KeyCode::Esc => {
                if self.busy() {
                    self.cancel.cancel();
                    self.toast("正在打断…");
                } else if !self.input.is_empty() {
                    self.input.take();
                }
            }
            KeyCode::Char('o') if ctrl => {
                self.expand_tools = !self.expand_tools;
                self.toast(if self.expand_tools {
                    "工具输出：展开"
                } else {
                    "工具输出：折叠"
                });
            }
            KeyCode::Char('t') if ctrl => {
                self.show_thinking = !self.show_thinking;
                self.toast(if self.show_thinking {
                    "思考过程：显示"
                } else {
                    "思考过程：隐藏"
                });
            }
            KeyCode::Char('r') if ctrl => self.open_sessions(false),
            KeyCode::Char('l') if ctrl => {
                self.entries.clear();
                self.scroll = 0;
            }
            KeyCode::BackTab => self.cycle_thinking(),
            KeyCode::PageUp => self.scroll_by(10),
            KeyCode::PageDown => self.scroll_by(-10),
            KeyCode::Up if ctrl || alt => self.scroll_by(1),
            KeyCode::Down if ctrl || alt => self.scroll_by(-1),
            KeyCode::Home if ctrl => self.scroll = self.max_scroll,
            KeyCode::End if ctrl => self.scroll = 0,
            KeyCode::Enter if shift => self.input.insert('\n'),
            KeyCode::Enter if alt => {
                // Alt+Enter: a follow-up that waits for the whole task.
                let text = self.input.take();
                self.submit_follow_up(text);
            }
            KeyCode::Char('j') if ctrl => self.input.insert('\n'),
            KeyCode::Char('v') if ctrl => self.paste_clipboard(),
            KeyCode::Char('y') if ctrl => self.copy_last_reply(),
            KeyCode::Char('g') if ctrl => self.open_agents(),
            KeyCode::Enter => {
                let matches = self.command_matches();
                if !matches.is_empty() {
                    let (name, args, _) = &matches[self.cmd_selected.min(matches.len() - 1)];
                    let exact = self.input.text[1..] == **name;
                    if !exact || !args.is_empty() && !args.starts_with('[') {
                        // complete the command first
                        self.input
                            .set(format!("/{name}{}", if args.is_empty() { "" } else { " " }));
                        if args.is_empty() || args.starts_with('[') {
                            let text = self.input.take();
                            self.submit(text);
                        }
                        return;
                    }
                }
                let text = self.input.take();
                self.submit(text);
            }
            KeyCode::Tab => {
                let matches = self.command_matches();
                if let Some((name, args, _)) =
                    matches.get(self.cmd_selected.min(matches.len().saturating_sub(1)))
                {
                    self.input
                        .set(format!("/{name}{}", if args.is_empty() { "" } else { " " }));
                }
            }
            KeyCode::Up => {
                let n = self.command_matches().len();
                if n > 0 {
                    self.cmd_selected = (self.cmd_selected + n - 1) % n;
                } else if self.input.on_first_line() {
                    if self.input.is_empty()
                        && let Some(t) = self.steer.pop_last()
                    {
                        self.input.set(t);
                    } else {
                        self.input.history_prev();
                    }
                } else {
                    self.input.up();
                }
            }
            KeyCode::Down => {
                let n = self.command_matches().len();
                if n > 0 {
                    self.cmd_selected = (self.cmd_selected + 1) % n;
                } else if self.input.on_last_line() {
                    self.input.history_next();
                } else {
                    self.input.down();
                }
            }
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Char('u') if ctrl => self.input.kill_line(),
            KeyCode::Char('w') if ctrl => self.input.delete_word(),
            KeyCode::Backspace if alt || ctrl => self.input.delete_word(),
            KeyCode::Backspace => {
                // A collapsed paste marker goes away in one keystroke.
                let byte = self
                    .input
                    .text
                    .char_indices()
                    .nth(self.input.cursor)
                    .map_or(self.input.text.len(), |(b, _)| b);
                let text = self.input.text.clone();
                if let Some((start, end)) = self.paste.marker_ending_at(&text, byte) {
                    let removed = text[start..end].chars().count();
                    self.input.text.replace_range(start..end, "");
                    self.input.cursor -= removed;
                } else {
                    self.input.backspace();
                }
            }
            KeyCode::Delete => self.input.delete(),
            KeyCode::Char(c) if !ctrl => {
                self.input.insert(c);
                self.cmd_selected = 0;
                if self.input.text == "/" {
                    // The command menu just opened: pick up skills installed
                    // while Limbo is running.
                    self.refresh_skills();
                }
            }
            _ => {}
        }
    }

    fn popup_key(&mut self, key: KeyEvent) {
        let Some(popup) = &mut self.popup else { return };
        match popup {
            Popup::Help => self.popup = None,
            Popup::VerifyPicker { items, selected } => match key.code {
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Down => *selected = (*selected + 1).min(items.len().saturating_sub(1)),
                KeyCode::Esc => {
                    self.popup = None;
                    self.skip_verify();
                }
                KeyCode::Enter => {
                    let choice = items[*selected].clone();
                    let first = items.first().filter(|c| *c != VERIFY_EDIT).cloned();
                    self.popup = None;
                    if choice == VERIFY_SKIP {
                        self.skip_verify();
                    } else if choice == VERIFY_EDIT {
                        self.editing_verify = true;
                        self.input.set(first.unwrap_or_default());
                        self.toast("编辑验收命令后回车确认（esc 取消）");
                    } else {
                        self.accept_verify(&choice);
                    }
                }
                _ => {}
            },
            Popup::Btw { scroll, .. } | Popup::Text { scroll, .. } => match key.code {
                KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
                KeyCode::Down | KeyCode::Char('j') => *scroll += 1,
                KeyCode::PageUp => *scroll = scroll.saturating_sub(10),
                KeyCode::PageDown | KeyCode::Char(' ') => *scroll += 10,
                _ => {
                    // Closing /btw discards the answer (and its stream).
                    self.btw_rx = None;
                    self.popup = None;
                }
            },
            Popup::Game(game) => match key.code {
                KeyCode::Up | KeyCode::Char('w') => {
                    game.slide(Dir::Up);
                }
                KeyCode::Down | KeyCode::Char('s') => {
                    game.slide(Dir::Down);
                }
                KeyCode::Left | KeyCode::Char('a') => {
                    game.slide(Dir::Left);
                }
                KeyCode::Right | KeyCode::Char('d') => {
                    game.slide(Dir::Right);
                }
                KeyCode::Char('r') => **game = Game::seeded(),
                KeyCode::Esc | KeyCode::Char('q') => self.popup = None,
                _ => {}
            },
            Popup::Agents { selected, scroll } => {
                let tasks = self.sup.as_ref().map(|s| s.tasks()).unwrap_or_default();
                match key.code {
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        *scroll = 0;
                    }
                    KeyCode::Down => {
                        *selected = (*selected + 1).min(tasks.len().saturating_sub(1));
                        *scroll = 0;
                    }
                    KeyCode::PageUp => *scroll = scroll.saturating_sub(10),
                    KeyCode::PageDown => *scroll += 10,
                    KeyCode::Char(c @ ('a' | 'x' | 'p' | 'r' | 'y' | 'n')) => {
                        if let (Some(sup), Some(t)) = (&self.sup, tasks.get(*selected)) {
                            let action = match c {
                                'a' | 'y' => "approve",
                                'n' => "reject",
                                'x' => "cancel",
                                'p' if t.status == "paused" => "resume",
                                'p' => "pause",
                                _ => "retry",
                            };
                            let msg =
                                sup.control(&t.id, action, "", None, false, None, None, "user");
                            self.toast(msg);
                        }
                    }
                    KeyCode::Char('@') => {
                        if let Some(t) = tasks.get(*selected) {
                            self.input.set(format!("@{} ", t.id));
                            self.popup = None;
                        }
                    }
                    _ => self.popup = None,
                }
            }
            Popup::Sessions {
                items,
                selected,
                all,
            } => match key.code {
                KeyCode::Esc => self.popup = None,
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Down => *selected = (*selected + 1).min(items.len().saturating_sub(1)),
                KeyCode::Tab => {
                    let all = !*all;
                    self.open_sessions(all);
                }
                KeyCode::Enter => {
                    if let Some(path) = items.get(*selected).and_then(|m| m.path.clone()) {
                        self.popup = None;
                        self.switch_session(&path);
                    }
                }
                _ => {}
            },
            Popup::Auth {
                provider,
                model,
                input,
                status,
                checking,
            } => match key.code {
                KeyCode::Esc => self.popup = None,
                _ if *checking => {}
                KeyCode::Backspace => {
                    input.pop();
                }
                KeyCode::Char(c) => input.push(c),
                KeyCode::Enter => {
                    let key = input.trim().to_string();
                    if key.is_empty() {
                        *status = Some("先粘贴一个 key".into());
                    } else {
                        *checking = true;
                        *status = Some("验证中…".into());
                        let (provider, model) = (provider.clone(), model.clone());
                        tokio::spawn(async move {
                            let Some(p) = crate::cli_auth::providers()
                                .into_iter()
                                .find(|p| p.id == provider)
                            else {
                                return;
                            };
                            let (detail, error) = match crate::cli_auth::check_key(&p, &key, None)
                                .await
                            {
                                crate::cli_auth::KeyCheck::Valid => ("key 有效".to_string(), false),
                                crate::cli_auth::KeyCheck::Unverified(d) => (d, false),
                                crate::cli_auth::KeyCheck::Invalid(d) => (d, true),
                            };
                            crate::background::emit(BackgroundEvent::AuthChecked {
                                provider,
                                model,
                                key,
                                detail,
                                error,
                            });
                        });
                    }
                }
                _ => {}
            },
            Popup::Models {
                items,
                selected,
                filter,
            } => match key.code {
                KeyCode::Esc => self.popup = None,
                KeyCode::Up => *selected = selected.saturating_sub(1),
                KeyCode::Down => *selected += 1,
                KeyCode::Backspace => {
                    filter.pop();
                    *selected = 0;
                }
                KeyCode::Char(c) => {
                    filter.push(c);
                    *selected = 0;
                }
                KeyCode::Enter => {
                    let visible: Vec<&(String, String)> = items
                        .iter()
                        .filter(|(id, _)| id.contains(filter.as_str()))
                        .collect();
                    let choice = visible
                        .get((*selected).min(visible.len().saturating_sub(1)))
                        .map(|(id, _)| id.clone());
                    let model = choice.unwrap_or_else(|| filter.clone());
                    self.popup = None;
                    if !model.is_empty() {
                        let spec = catalog::resolve_model(&model);
                        if catalog::resolve_api_key(&spec, &self.config).is_none() {
                            // No key yet: log in first, then switch.
                            self.popup = Some(Popup::Auth {
                                provider: spec.provider.id.to_string(),
                                model,
                                input: String::new(),
                                status: catalog::resolve_api_key_env(&spec, &self.config)
                                    .map(|e| format!("也可以设置环境变量 ${e}")),
                                checking: false,
                            });
                            self.fx.popup_opened();
                        } else {
                            self.set_model(&model);
                        }
                    }
                }
                _ => {}
            },
        }
    }

    // -- commands ----------------------------------------------------------------

    fn submit(&mut self, text: String) {
        let trimmed = text.trim().to_string();
        if self.editing_verify {
            self.editing_verify = false;
            self.paste.clear();
            if trimmed.is_empty() {
                self.skip_verify();
            } else {
                self.accept_verify(&trimmed);
            }
            return;
        }
        if trimmed.is_empty() {
            return;
        }
        self.cmd_selected = 0;
        self.queue_held = false;
        if let Some(rest) = trimmed.strip_prefix('/')
            && !rest.starts_with('/')
            && !rest.starts_with(' ')
            && !text.starts_with(' ')
        {
            let (cmd, arg) = rest
                .split_once(char::is_whitespace)
                .map_or((rest, ""), |(c, a)| (c, a.trim()));
            let (cmd, arg) = (cmd.to_string(), arg.to_string());
            if self.run_command(&cmd, &arg) {
                self.paste.clear();
                return;
            }
            if !self.skills.iter().any(|s| s.name == cmd) {
                self.refresh_skills();
            }
            if let Some(skill) = self.skills.iter().find(|s| s.name == cmd).cloned() {
                let mut prompt = format!(
                    "# Skill: {}\n\n{}\n\n(Skill 文件位于 {}，其中引用的相对路径基于其所在目录解析。)",
                    skill.name,
                    skill.body.trim(),
                    skill.path.display()
                );
                if !arg.is_empty() {
                    let (expanded, _) = self.paste.expand(&arg);
                    prompt.push_str(&format!("\n\n## 用户输入\n\n{expanded}"));
                }
                let atts = self.paste.live_attachments(&arg);
                self.paste.clear();
                return self.send_prompt(trimmed, prompt, atts, STEER);
            }
            // Not a command: a path like /etc/hosts is sent as text.
            if !paste::looks_like_path(&trimmed) {
                self.toast(format!("未知命令 /{cmd}，输入 /help 查看命令"));
                return;
            }
        }
        // @name talks to one subagent directly.
        if let Some(sup) = &self.sup
            && let Some(reply) = crate::subagents::direct_message(sup, &trimmed)
        {
            self.paste.clear();
            self.notice(NoticeKind::Info, format!("➤ {reply}"));
            return;
        }
        let (prompt, atts) = self.take_message(&text);
        self.send_prompt(text, prompt, atts, STEER);
    }

    /// Expand collapsed pastes and collect the message's attachments.
    fn take_message(&mut self, text: &str) -> (String, Vec<Attachment>) {
        let (expanded, invalid) = self.paste.expand(text);
        if !invalid.is_empty() {
            self.toast("有粘贴内容已失效，按原样发送了占位符");
        }
        let mut atts = self.paste.live_attachments(text);
        for a in paste::attachments_from_text(&expanded) {
            if !atts.iter().any(|x| x.path == a.path) {
                atts.push(a);
            }
        }
        self.paste.clear();
        (expanded, atts)
    }

    /// Alt+Enter / `/queue`: wait for the whole task, not just the next step.
    fn submit_follow_up(&mut self, text: String) {
        if text.trim().is_empty() {
            return;
        }
        if !self.busy() {
            return self.submit(text);
        }
        let (prompt, atts) = self.take_message(&text);
        self.steer
            .push_item(prompt, atts, FOLLOW_UP, "user", vec![]);
        self.toast("已排队，当前任务全部完成后发送（↑ 撤回）");
    }

    fn send_prompt(&mut self, display: String, prompt: String, atts: Vec<Attachment>, mode: &str) {
        if self.busy() {
            self.steer.push_item(prompt, atts, mode, "user", vec![]);
            self.toast("已排队，将在当前步骤后发送（↑ 撤回）");
            return;
        }
        let shown = if atts.is_empty() {
            display
        } else {
            format!(
                "{display}\n📎 {}",
                atts.iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>()
                    .join(" · ")
            )
        };
        self.entries
            .push(Entry::new(EntryKind::User { text: shown }));
        self.start_job(Job::Turn(prompt, atts));
    }

    pub fn open_agents(&mut self) {
        if self.sup.is_none() {
            self.toast("子代理已关闭（[subagents] enabled = false）");
            return;
        }
        self.popup = Some(Popup::Agents {
            selected: 0,
            scroll: 0,
        });
        self.fx.popup_opened();
    }

    fn text_popup(&mut self, title: &str, body: String) {
        self.popup = Some(Popup::Text {
            title: title.into(),
            body,
            scroll: 0,
        });
        self.fx.popup_opened();
    }

    /// Returns true when `cmd` was a built-in command.
    fn run_command(&mut self, cmd: &str, arg: &str) -> bool {
        let needs_agent = matches!(
            cmd,
            "new"
                | "fork"
                | "handoff"
                | "compact"
                | "model"
                | "thinking"
                | "plan"
                | "export"
                | "sessions"
                | "goal"
                | "code-mode"
                | "daily-mode"
                | "memory"
                | "learn"
        );
        if needs_agent && self.busy() {
            if COMMANDS.iter().any(|c| c.name == cmd) {
                self.toast("正在运行，请先等待或按 Esc 打断");
                return true;
            }
            return false;
        }
        match cmd {
            "help" | "?" => self.popup = Some(Popup::Help),
            "quit" | "exit" | "q" => self.quit = true,
            "clear" => {
                self.entries.clear();
                self.scroll = 0;
            }
            "mouse" => {
                self.mouse_capture = !self.mouse_capture;
                self.mouse_changed = true;
                self.toast(if self.mouse_capture {
                    "鼠标捕获：开（滚轮滚动）"
                } else {
                    "鼠标捕获：关（可直接选择复制）"
                });
            }
            "theme" => {
                let next = if arg.is_empty() {
                    let i = THEMES
                        .iter()
                        .position(|t| t.name == self.theme.name)
                        .unwrap_or(0);
                    THEMES[(i + 1) % THEMES.len()]
                } else {
                    theme::by_name(Some(arg))
                };
                self.theme = next;
                for e in &mut self.entries {
                    e.touch();
                }
                self.fx.theme_changed(&self.theme);
                self.toast(format!("主题：{}", next.name));
            }
            "new" => {
                if let Some(a) = &mut self.agent {
                    a.new_session();
                    self.sup = a.supervisor.clone();
                }
                self.entries.clear();
                self.refresh_info();
                self.fx.startup(&self.theme);
            }
            "sessions" | "resume" => self.open_sessions(false),
            "compact" => self.start_job(Job::Compact),
            "plan" => {
                if let Some(a) = &mut self.agent {
                    let on = match arg {
                        "on" => true,
                        "off" => false,
                        _ => !a.plan_mode(),
                    };
                    a.set_plan_mode(on);
                    self.notice(
                        NoticeKind::Info,
                        if on {
                            "计划模式：开（只读，结束时给出计划；/plan off 退出）"
                        } else {
                            "计划模式：关"
                        },
                    );
                }
                self.refresh_info();
            }
            "model" => {
                if arg.is_empty() {
                    self.open_models();
                } else {
                    let (model, level) = catalog::split_model_arg(arg);
                    self.set_model(&model);
                    if let (Some(a), Some(l)) = (&mut self.agent, level) {
                        a.set_thinking(Some(l));
                        self.refresh_info();
                    }
                }
            }
            "thinking" => {
                if arg.is_empty() {
                    self.cycle_thinking();
                } else if let Some(a) = &mut self.agent {
                    a.set_thinking(if arg == "default" {
                        None
                    } else {
                        Some(arg.to_string())
                    });
                    self.refresh_info();
                    self.toast(format!("思考强度：{arg}"));
                }
            }
            "export" => {
                if let Some(a) = &self.agent {
                    let path = if arg.is_empty() {
                        a.workdir.join(format!("limbo-{}.md", a.session_id()))
                    } else {
                        crate::tools::resolve_path(&a.workdir, arg)
                    };
                    match session::export_markdown(&a.meta, &a.messages, &path) {
                        Ok(()) => {
                            self.notice(NoticeKind::Info, format!("已导出到 {}", path.display()))
                        }
                        Err(e) => self.notice(NoticeKind::Error, format!("导出失败：{e}")),
                    }
                }
            }
            "fork" => {
                let keep = arg.parse::<usize>().ok();
                let result = self.agent.as_mut().map(|a| a.fork(keep));
                match result {
                    Some(Ok(path)) => {
                        self.switch_session(&path);
                        self.notice(NoticeKind::Info, "已分叉到新会话，原会话保持不变");
                    }
                    Some(Err(e)) => self.notice(NoticeKind::Error, e.to_string()),
                    None => {}
                }
            }
            "handoff" => {
                if arg.is_empty() {
                    self.toast("用法：/handoff <交接说明>");
                    return true;
                }
                let mut prompt = String::new();
                if let Some(a) = &mut self.agent {
                    prompt = a.handoff_prompt(arg);
                    let meta = a.handoff_meta();
                    a.reset_to(meta);
                    self.sup = a.supervisor.clone();
                }
                self.entries.clear();
                self.refresh_info();
                self.notice(NoticeKind::Info, "已交接到新会话");
                self.send_prompt(format!("/handoff {arg}"), prompt, vec![], STEER);
            }
            "queue" => {
                if arg.is_empty() {
                    self.toast("用法：/queue <文本>");
                } else {
                    self.submit_follow_up(arg.to_string());
                }
            }
            "goal" => self.cmd_goal(arg),
            "code-mode" => {
                let Some(a) = self.agent.as_mut() else {
                    return true;
                };
                match parse_toggle(arg, a.code_mode()) {
                    None => self.toast(cmds::CODE_MODE_USAGE),
                    Some(on) if a.set_code_mode(on) => {
                        self.notice(NoticeKind::Info, cmds::code_mode_notice(on))
                    }
                    Some(on) => self.toast(cmds::unchanged_notice("Code mode ", on)),
                }
                self.refresh_info();
            }
            "daily-mode" => {
                let Some(a) = self.agent.as_mut() else {
                    return true;
                };
                match parse_toggle(arg, a.daily_mode()) {
                    None => self.toast(cmds::DAILY_USAGE),
                    Some(on) if a.set_daily_mode(on) => {
                        self.skills = discover_skills(&a.workdir, on);
                        self.notice(
                            NoticeKind::Info,
                            if on {
                                cmds::DAILY_DIVIDER_ON
                            } else {
                                cmds::DAILY_DIVIDER_OFF
                            },
                        );
                        self.notice(NoticeKind::Info, cmds::daily_toggle_notice(on));
                    }
                    Some(on) => self.toast(cmds::unchanged_notice("Daily mode ", on)),
                }
                self.refresh_info();
                self.apply_look();
            }
            "memory" | "learn" if !self.agent.as_ref().is_some_and(|a| a.daily_mode()) => {
                self.toast(cmds::daily_only_notice(&format!("/{cmd}")));
            }
            "memory" => {
                let Some(a) = self.agent.as_ref() else {
                    return true;
                };
                match crate::memory::parse_memory_command(arg) {
                    Err(()) => self.toast(crate::memory::MEMORY_USAGE),
                    Ok(target) => {
                        let store = a.memory_store();
                        match (target, &store) {
                            (Some(t), Some(st)) => {
                                let path = st.path(&t);
                                if let Some(parent) = path.parent() {
                                    let _ = std::fs::create_dir_all(parent);
                                }
                                self.pending_editor = Some(path);
                            }
                            _ => {
                                let body =
                                    crate::memory::describe(store.as_ref(), a.memory.as_ref());
                                self.text_popup("长期记忆", body);
                            }
                        }
                    }
                }
            }
            "learn" => {
                if let Some(a) = self.agent.as_mut() {
                    let msg = crate::learn::start_review(a, false);
                    self.notice(NoticeKind::Info, msg);
                }
            }
            "btw" => {
                if arg.is_empty() {
                    self.toast("用法：/btw <问题>");
                    return true;
                }
                let Some((client, history)) = self.snapshot() else {
                    return true;
                };
                self.btw_rx = Some(cmds::btw_query(&client, &history, arg));
                self.popup = Some(Popup::Btw {
                    question: arg.to_string(),
                    answer: String::new(),
                    done: false,
                    scroll: 0,
                });
                self.fx.popup_opened();
            }
            "agents" => match &self.sup {
                None => self.toast("子代理已关闭（[subagents] enabled = false）"),
                Some(_) if arg.is_empty() => self.open_agents(),
                Some(sup) => {
                    let body = crate::subagents::agents_command(sup, arg);
                    self.text_popup("子代理", body);
                }
            },
            "task" => match &self.sup {
                None => self.toast("子代理已关闭（[subagents] enabled = false）"),
                Some(sup) => {
                    let body = crate::subagents::task_command(sup, arg);
                    if body.lines().count() <= 2 {
                        self.notice(NoticeKind::Info, body);
                    } else {
                        self.text_popup("任务板", body);
                    }
                }
            },
            "tracing-viewer" => {
                let id = self.info.session_id.clone();
                let msg = self.viewer.command(arg, Some(&id));
                self.notice(NoticeKind::Info, msg);
            }
            "copy" => self.copy_last_reply(),
            "2048" => {
                self.popup = Some(Popup::Game(Box::new(Game::seeded())));
                self.fx.popup_opened();
            }
            _ => return false,
        }
        true
    }

    /// The model client and history a /btw side question runs over.
    fn snapshot(&self) -> Option<(crate::llm::LlmClient, Vec<crate::models::Message>)> {
        if let Some(a) = &self.agent {
            return Some((a.client.clone(), a.messages.clone()));
        }
        self.busy_snapshot.clone()
    }

    fn cmd_goal(&mut self, arg: &str) {
        let Some(agent) = self.agent.as_mut() else {
            return;
        };
        match parse_goal_args(arg) {
            GoalCommand::Query => match pump::live_goal(agent) {
                None => self.toast("当前没有 goal（/goal <目标> 开始）"),
                Some(g) => {
                    let body = cmds::goal_status_text(g);
                    self.text_popup("Goal", body);
                }
            },
            GoalCommand::Clear => {
                if pump::clear_goal(agent) {
                    self.notice(NoticeKind::Info, "已退出 goal 模式");
                } else {
                    self.toast("当前没有 goal");
                }
                self.refresh_info();
            }
            GoalCommand::Set(text) => {
                let prompt = pump::set_goal(agent, &text);
                self.proposing = true;
                self.entries.push(Entry::new(EntryKind::User {
                    text: format!("/goal {text}"),
                }));
                self.refresh_info();
                self.start_job(Job::Turn(prompt, vec![]));
            }
        }
    }

    fn switch_session(&mut self, path: &std::path::Path) {
        let Some(old) = &self.agent else { return };
        let dir = old.session_dir.clone();
        let mut cfg = self.config.clone();
        cfg.llm.model = old.client.model_id().to_string();
        match Agent::resume(cfg, path, &dir) {
            Ok(agent) => {
                if let Some(old) = self.agent.as_ref().and_then(|a| a.supervisor.clone()) {
                    old.stop();
                }
                self.steer = agent.steer.clone();
                self.sup = agent.supervisor.clone();
                self.entries = entries_from_history(&agent.messages);
                self.agent = Some(agent);
                self.scroll = 0;
                self.refresh_info();
                self.fx.startup(&self.theme);
            }
            Err(e) => self.notice(NoticeKind::Error, format!("无法打开会话：{e}")),
        }
    }

    fn open_sessions(&mut self, all: bool) {
        let Some(a) = &self.agent else {
            self.toast("正在运行，请稍后");
            return;
        };
        let items = session::list_sessions(&a.session_dir, (!all).then_some(a.workdir.as_path()));
        self.popup = Some(Popup::Sessions {
            items,
            selected: 0,
            all,
        });
        self.fx.popup_opened();
    }

    fn open_models(&mut self) {
        let recent = crate::recent_models::load();
        let label = |s: &catalog::ModelSpec, tag: &str| {
            let key = if catalog::resolve_api_key(s, &self.config).is_some() {
                "●"
            } else {
                "○"
            };
            (
                s.id.clone(),
                format!(
                    "{key} {}{tag}",
                    catalog::provider_display_name(s.provider.id)
                ),
            )
        };
        // Recently used models first, then the rest of the catalog.
        let mut items: Vec<(String, String)> = recent
            .iter()
            .filter_map(|id| catalog::catalog().get(id))
            .map(|s| label(s, " · 最近"))
            .collect();
        items.extend(
            catalog::catalog()
                .values()
                .filter(|s| !recent.contains(&s.id))
                .map(|s| label(s, "")),
        );
        let selected = items
            .iter()
            .position(|(id, _)| *id == self.info.model)
            .unwrap_or(0);
        self.popup = Some(Popup::Models {
            items,
            selected,
            filter: String::new(),
        });
        self.fx.popup_opened();
    }

    fn set_model(&mut self, model: &str) {
        let Some(a) = &mut self.agent else { return };
        let previous = a.client.spec.id.clone();
        match a.set_model(model) {
            Ok(()) => {
                crate::recent_models::push(&previous);
                crate::recent_models::push(model);
                let saved = config::save_model_to_config(model, None).is_ok();
                self.config.llm.model = model.to_string();
                self.refresh_info();
                self.toast(format!(
                    "模型 → {model}{}",
                    if saved {
                        "（已保存到配置）"
                    } else {
                        ""
                    }
                ));
            }
            Err(e) => self.notice(NoticeKind::Error, format!("{e}")),
        }
    }

    fn cycle_thinking(&mut self) {
        let Some(a) = &mut self.agent else { return };
        let choices = catalog::thinking_choices(&a.client.spec);
        if choices.is_empty() {
            self.toast("当前模型没有思考强度开关");
            return;
        }
        let cur = a.client.thinking_effort.clone();
        let i = choices.iter().position(|c| *c == cur).unwrap_or(0);
        let next = choices[(i + 1) % choices.len()].clone();
        a.set_thinking(next.clone());
        let _ = config::save_thinking_to_config(next.as_deref(), None);
        self.config.llm.thinking_effort = next.clone();
        self.refresh_info();
        self.toast(format!(
            "思考强度：{}",
            next.unwrap_or_else(|| "默认".into())
        ));
    }

    pub fn interrupt(&self) {
        self.cancel.cancel();
    }

    /// Persist on exit.
    pub fn shutdown(&mut self) {
        if let Some(a) = &mut self.agent {
            a.checkpoint();
            a.trace.flush();
        }
        if let Some(sup) = &self.sup {
            sup.stop();
        }
        self.viewer.stop();
        self.reporters.release();
    }
}

pub fn fmt_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}
