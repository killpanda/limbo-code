//! TUI application state: keys, slash commands, agent events.

use std::sync::Arc;
use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::chat::{Entry, EntryKind, NoticeKind, ToolStatus, entries_from_history};
use super::fx::Fx;
use super::input::Input;
use super::theme::{self, THEMES, Theme};
use crate::agent::{Agent, AgentEvent, EventTx, SteerQueue};
use crate::config::{self, Config};
use crate::llm::catalog;
use crate::skills::{Skill, discover_skills};
use crate::session::{self, SessionMeta};

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
}

pub enum Job {
    Turn(String),
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
        name: "export",
        args: "[path]",
        help: "导出对话为 Markdown",
    },
    SlashCommand {
        name: "theme",
        args: "[neon|aurora|ember]",
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
    pub fx: Fx,
    pub quit: bool,
    last_ctrl_c: Option<Instant>,
    pub mouse_capture: bool,
    pub mouse_changed: bool,
    pub toast: Option<(String, Instant)>,
    pub skills: Vec<Skill>,
    pub dirty: bool,
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
        let mut app = Self {
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
            quit: false,
            last_ctrl_c: None,
            mouse_capture: true,
            mouse_changed: false,
            toast: None,
            skills,
            dirty: true,
        };
        app.refresh_info();
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
        self.steer.texts()
    }

    fn refresh_info(&mut self) {
        if let Some(a) = &self.agent {
            self.info = Info {
                model: a.client.model_id().to_string(),
                session_id: a.session_id().to_string(),
                title: a.meta.title.clone(),
                context_tokens: a.context_tokens(),
                context_window: a.context_window(),
                session_total: a.session_total_tokens(),
                plan_mode: a.plan_mode(),
                thinking: a.client.thinking_effort.clone(),
                workdir: a.workdir.display().to_string(),
            };
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
        self.activity = if matches!(job, Job::Compact) {
            Activity::Compacting
        } else {
            Activity::Thinking
        };
        self.scroll = 0;
        // The agent moves into the task and comes back when it finishes.
        self.job = Some(tokio::spawn(async move {
            match job {
                Job::Turn(text) => agent.run(text, vec![], &tx, &cancel).await,
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
        // Something was queued after the turn's last drain: keep going.
        if !self.steer.is_empty() {
            self.start_job(Job::Turn(String::new()));
        }
        self.dirty = true;
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
            AgentEvent::Warning(t) => self.notice(NoticeKind::Warn, t),
            AgentEvent::Error(t) => {
                self.notice(NoticeKind::Error, t);
                self.fx.error(&self.theme);
            }
            AgentEvent::Interrupted => self.notice(NoticeKind::Warn, "已打断"),
            AgentEvent::TurnEnd => self.close_live(),
        }
    }

    fn find_tool(&mut self, id: &str) -> Option<&mut Entry> {
        self.entries
            .iter_mut()
            .rev()
            .find(|e| matches!(&e.kind, EntryKind::Tool { id: i, .. } if i == id))
    }

    // -- input ----------------------------------------------------------------

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
        self.input.insert_str(&text);
        self.dirty = true;
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
            KeyCode::Enter if shift || alt => self.input.insert('\n'),
            KeyCode::Char('j') if ctrl => self.input.insert('\n'),
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
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Char(c) if !ctrl => {
                self.input.insert(c);
                self.cmd_selected = 0;
            }
            _ => {}
        }
    }

    fn popup_key(&mut self, key: KeyEvent) {
        let Some(popup) = &mut self.popup else { return };
        match popup {
            Popup::Help => self.popup = None,
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
                        self.set_model(&model);
                    }
                }
                _ => {}
            },
        }
    }

    // -- commands ----------------------------------------------------------------

    fn submit(&mut self, text: String) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }
        self.cmd_selected = 0;
        if let Some(rest) = trimmed.strip_prefix('/')
            && !rest.starts_with('/')
            && !rest.starts_with(' ')
        {
            let (cmd, arg) = rest
                .split_once(char::is_whitespace)
                .map_or((rest, ""), |(c, a)| (c, a.trim()));
            if self.run_command(cmd, arg) {
                return;
            }
            if let Some(skill) = self.skills.iter().find(|s| s.name == cmd).cloned() {
                let mut prompt = format!(
                    "# Skill: {}\n\n{}\n\n(Skill 文件位于 {}，其中引用的相对路径基于其所在目录解析。)",
                    skill.name,
                    skill.body.trim(),
                    skill.path.display()
                );
                if !arg.is_empty() {
                    prompt.push_str(&format!("\n\n## 用户输入\n\n{arg}"));
                }
                return self.send_prompt(trimmed.to_string(), prompt);
            }
            // Not a command: a path like /etc/hosts is sent as text.
            if !std::path::Path::new(cmd).exists() && !cmd.contains('/') {
                self.toast(format!("未知命令 /{cmd}，输入 /help 查看命令"));
                return;
            }
        }
        self.send_prompt(text.clone(), text);
    }

    fn send_prompt(&mut self, display: String, prompt: String) {
        if self.busy() {
            self.steer.push(prompt);
            self.toast("已排队，将在当前步骤后发送（↑ 撤回）");
            return;
        }
        self.entries
            .push(Entry::new(EntryKind::User { text: display }));
        self.start_job(Job::Turn(prompt));
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
                }
                self.entries.clear();
                self.refresh_info();
                self.notice(NoticeKind::Info, "已交接到新会话");
                self.send_prompt(format!("/handoff {arg}"), prompt);
            }
            _ => return false,
        }
        true
    }

    fn switch_session(&mut self, path: &std::path::Path) {
        let Some(old) = &self.agent else { return };
        let dir = old.session_dir.clone();
        let mut cfg = self.config.clone();
        cfg.llm.model = old.client.model_id().to_string();
        match Agent::resume(cfg, path, &dir) {
            Ok(agent) => {
                self.steer = agent.steer.clone();
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
        let items: Vec<(String, String)> = catalog::catalog()
            .values()
            .map(|s| {
                let key = if catalog::resolve_api_key(s, &self.config).is_some() {
                    "●"
                } else {
                    "○"
                };
                (
                    s.id.clone(),
                    format!("{key} {}", catalog::provider_display_name(s.provider.id)),
                )
            })
            .collect();
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
        match a.set_model(model) {
            Ok(()) => {
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
        }
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
