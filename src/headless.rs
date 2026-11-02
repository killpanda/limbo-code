//! Headless frontend: a line-oriented REPL on plain stdout (for scripts and
//! terminal multiplexers such as herdr), plus `--json` (one JSON object per
//! event) and `-p` one-shot mode.
//!
//! stdin is read concurrently with the running task, so a line typed while
//! the agent works is *steered* into the turn (`/queue <text>` waits for the
//! whole task instead). All orchestration — follow-ups, the goal loop, queue
//! policy — lives in [`crate::pump`]; this module only translates lines into
//! calls and events into text.

use crate::integrations::{Reporters, State, TurnOutcome};
use std::io::Write;
use std::sync::Arc;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEvent, FOLLOW_UP, STEER, SteerQueue};
use crate::background::BackgroundEvent;
use crate::commands::{self as cmds, parse_toggle};
use crate::goal::{GoalCommand, parse_goal_args};
use crate::pump;
use crate::subagents::describe_call;

pub const HELP_TEXT: &str = "可用命令：/goal <目标>|clear 目标闭环 · /queue <文本> 任务完成后再发 · /compact 压缩上下文 · \
/model [id] 切换模型 · /new 新会话 · /fork [n] 分叉 · /handoff <说明> 交接 · /plan on|off 计划模式 · \
/code-mode on|off · /daily-mode on|off · /btw <问题> 侧问 · /agents [scan|init] 子代理 · /task 任务板 · @名字 <消息> · /tracing-viewer [id|off] · /help · /quit";
pub const DAILY_HELP_TEXT: &str = " · /memory [edit] 查看记忆 · /learn 复盘写进 skill";

fn event_json(ev: &AgentEvent) -> serde_json::Value {
    match ev {
        AgentEvent::TurnStart => json!({"type": "turn_start"}),
        AgentEvent::Text(t) => json!({"type": "text", "text": t}),
        AgentEvent::Thinking(t) => json!({"type": "thinking", "text": t}),
        AgentEvent::Retry { attempt, reason } => {
            json!({"type": "retry", "attempt": attempt, "reason": reason})
        }
        AgentEvent::ToolCall {
            id,
            name,
            arguments,
        } => json!({"type": "tool_call", "id": id, "name": name, "arguments": arguments}),
        AgentEvent::ToolProgress { id, line } => {
            json!({"type": "tool_progress", "id": id, "line": line})
        }
        AgentEvent::ToolResult {
            id,
            name,
            success,
            output,
        } => {
            json!({"type": "tool_result", "id": id, "name": name, "success": success, "output": output})
        }
        AgentEvent::Compaction {
            compacted,
            trigger,
            reason,
            tokens_before,
            tokens_after,
        } => json!({
            "type": "compaction", "compacted": compacted, "trigger": trigger, "reason": reason,
            "tokens_before": tokens_before, "tokens_after": tokens_after,
        }),
        AgentEvent::Usage {
            context_tokens,
            context_window,
            session_total,
        } => {
            json!({"type": "usage", "context_tokens": context_tokens, "context_window": context_window, "session_total": session_total})
        }
        AgentEvent::Steer(t) => json!({"type": "steer", "text": t}),
        AgentEvent::Info(t) => json!({"type": "info", "text": t}),
        AgentEvent::Warning(t) => json!({"type": "warning", "message": t}),
        AgentEvent::Error(t) => json!({"type": "error", "message": t}),
        AgentEvent::Interrupted => json!({"type": "interrupted"}),
        AgentEvent::TurnEnd => json!({"type": "turn_end"}),
        AgentEvent::GoalVerifyStarted {
            command,
            round,
            max_rounds,
        } => {
            json!({"type": "goal_verify_started", "command": command, "round": round, "max_rounds": max_rounds})
        }
        AgentEvent::GoalVerifyResult {
            exit_code,
            output_tail,
            timed_out,
            cancelled,
            round,
            max_rounds,
        } => {
            json!({"type": "goal_verify_result", "exit_code": exit_code, "output_tail": output_tail,
                     "timed_out": timed_out, "cancelled": cancelled, "round": round, "max_rounds": max_rounds})
        }
        AgentEvent::GoalPassed { rounds } => json!({"type": "goal_passed", "rounds": rounds}),
        AgentEvent::GoalExhausted { rounds } => json!({"type": "goal_exhausted", "rounds": rounds}),
        AgentEvent::GoalResumed { max_rounds } => {
            json!({"type": "goal_resumed", "max_rounds": max_rounds})
        }
        AgentEvent::QueueHeld { policy, texts } => {
            json!({"type": "queue_held", "policy": policy, "texts": texts})
        }
    }
}

struct Printer {
    json: bool,
    in_text: bool,
    total_tokens: u64,
}

impl Printer {
    fn line(&mut self, text: &str) {
        let mut out = std::io::stdout();
        if self.json {
            let _ = writeln!(out, "{}", json!({"type": "info", "text": text}));
        } else {
            self.end_text();
            let _ = writeln!(out, "{text}");
        }
        let _ = out.flush();
    }

    fn end_text(&mut self) {
        if self.in_text {
            println!();
            self.in_text = false;
        }
    }

    fn event(&mut self, ev: &AgentEvent) {
        let mut out = std::io::stdout();
        if let AgentEvent::Usage { session_total, .. } = ev {
            self.total_tokens = *session_total;
        }
        if self.json {
            let _ = writeln!(out, "{}", event_json(ev));
            let _ = out.flush();
            return;
        }
        if let Some(line) = cmds::goal_event_line(ev) {
            return self.line(&line);
        }
        match ev {
            AgentEvent::Text(t) => {
                self.in_text = true;
                let _ = write!(out, "{t}");
            }
            AgentEvent::Thinking(_) | AgentEvent::TurnStart | AgentEvent::Usage { .. } => {}
            AgentEvent::ToolCall {
                name, arguments, ..
            } => {
                let desc = if name == "run_code" {
                    arguments
                        .get("description")
                        .and_then(|d| d.as_str())
                        .map(|d| format!("run_code · {}", d.lines().next().unwrap_or("")))
                        .unwrap_or_else(|| describe_call(name, arguments))
                } else {
                    describe_call(name, arguments)
                };
                self.line(&format!("⚙ {desc}"));
            }
            AgentEvent::ToolProgress { line, .. } => self.line(&format!("  │ {line}")),
            AgentEvent::ToolResult {
                success, output, ..
            } => {
                let first = output.lines().next().unwrap_or("");
                self.line(&format!(
                    "  {} {}",
                    if *success { "✓" } else { "✗" },
                    crate::llm::truncate_chars(first, 120)
                ));
            }
            AgentEvent::Retry { attempt, reason } => self.line(&format!(
                "⟳ 连接中断，重试（{attempt}）：{reason}；以上半段回复作废"
            )),
            AgentEvent::Compaction {
                compacted,
                trigger,
                reason,
                tokens_before,
                tokens_after,
            } => {
                if *compacted {
                    self.line(&format!(
                        "已压缩上下文：约 {tokens_before} → {tokens_after} tokens{}",
                        if trigger == "manual" {
                            "（手动）"
                        } else {
                            "（自动）"
                        }
                    ));
                } else if let Some(r) = reason {
                    self.line(r);
                }
            }
            AgentEvent::Steer(t) => {
                self.line(&format!("➤ 插话已注入：{}", t.lines().next().unwrap_or("")))
            }
            AgentEvent::Info(t) => self.line(t),
            AgentEvent::Warning(t) => self.line(&format!("⚠ {t}")),
            AgentEvent::Error(t) => self.line(&format!("✗ {t}")),
            AgentEvent::Interrupted => self.line("⏹ 已打断"),
            AgentEvent::TurnEnd => self.end_text(),
            _ => {}
        }
        let _ = out.flush();
    }

    fn idle(&mut self) {
        if self.json {
            println!(
                "{}",
                json!({"type": "idle", "total_tokens": self.total_tokens})
            );
        } else {
            let t = self.total_tokens;
            self.line(&format!("── limbo idle · 累计 tokens {t} ──"));
        }
    }
}

/// One prompt, print the answer, exit (the goal loop / follow-ups still run).
pub async fn run_once(mut agent: Agent, prompt: String, json_mode: bool) -> i32 {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let c2 = cancel.clone();
    let ctrlc = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            c2.cancel();
        }
    });
    let printer = tokio::spawn(async move {
        let mut p = Printer {
            json: json_mode,
            in_text: false,
            total_tokens: 0,
        };
        while let Some(ev) = rx.recv().await {
            p.event(&ev);
        }
        p.end_text();
    });
    pump::run_task(&mut agent, prompt, vec![], &tx, &cancel).await;
    drop(tx);
    let _ = printer.await;
    ctrlc.abort();
    agent.checkpoint();
    if agent.last_run_failed { 1 } else { 0 }
}

enum Job {
    Task(String),
    Compact,
}

struct Repl {
    agent: Option<Agent>,
    job: Option<JoinHandle<Agent>>,
    cancel: CancellationToken,
    steer: Arc<SteerQueue>,
    tx: mpsc::UnboundedSender<AgentEvent>,
    p: Printer,
    /// The task that just ran was a /goal proposal round.
    proposing: bool,
    /// Blocked on the user's answer to a verify proposal.
    proposals: Option<Vec<String>>,
    quit: bool,
    eof: bool,
    eof_waiting: bool,
    viewer: crate::trace_viewer::Viewer,
    /// The board while the agent is away running a task.
    sup_handle: Option<std::sync::Arc<crate::subagents::Supervisor>>,
    /// Herdr / OSC 7501 lifecycle reports.
    reporters: std::sync::Arc<Reporters>,
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

impl Repl {
    fn busy(&self) -> bool {
        self.job.is_some()
    }

    fn start(&mut self, job: Job) {
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
                Job::Task(text) => pump::run_task(&mut agent, text, vec![], &tx, &cancel).await,
                Job::Compact => {
                    agent.compact("manual", &tx, &cancel).await;
                    agent.checkpoint();
                }
            }
            agent
        }));
    }

    fn job_done(&mut self, agent: Agent) {
        self.sup_handle = agent.supervisor.clone();
        self.agent = Some(agent);
        self.job = None;
        self.p.idle();
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
        self.finish_if_eof();
    }

    /// After stdin closed: quit once the main agent is idle and no subagent
    /// is still working (their results would otherwise be lost).
    fn finish_if_eof(&mut self) {
        if !self.eof || self.busy() || self.quit {
            return;
        }
        if self.proposals.take().is_some() {
            self.p.line("输入已关闭，跳过验收确认（保持单轮模式）");
        }
        // Results that already arrived go to the main agent first.
        self.maybe_wake_main();
        if self.busy() {
            return;
        }
        if self.sup().is_some_and(|s| s.working()) {
            if !std::mem::replace(&mut self.eof_waiting, true) {
                self.p.line("输入已关闭，等待子代理完成…（Ctrl+C 退出）");
            }
            return;
        }
        self.quit = true;
    }

    fn confirm_proposal(&mut self) {
        let Some(agent) = &self.agent else { return };
        if pump::live_goal(agent).is_none() {
            return;
        }
        match pump::take_proposal(agent) {
            Some(cmds_) if cmds_.is_empty() => self.p.line(cmds::NO_VERIFY_GATE),
            found => {
                let list = found.unwrap_or_default();
                if list.is_empty() {
                    self.p.line("未能解析模型的验收提议。");
                } else {
                    self.p.line("模型提议的验收命令：");
                    for (i, c) in list.iter().enumerate() {
                        self.p.line(&format!("  {}. {c}", i + 1));
                    }
                }
                self.p
                    .line("回复序号采纳；或直接输入一行作为验收命令；/skip 跳过自动验收");
                self.proposals = Some(list);
            }
        }
    }

    fn answer_verify(&mut self, text: &str, proposals: Vec<String>) {
        let t = text.trim();
        if t.is_empty() || t == "/skip" {
            self.p.line(cmds::VERIFY_SKIPPED);
            self.reporters.report(State::Idle, None);
            return;
        }
        if t.starts_with('/') {
            self.p
                .line("正在等待验收确认：回复序号、验收命令，或 /skip 跳过");
            self.proposals = Some(proposals);
            return;
        }
        let command = match t.parse::<usize>() {
            Ok(i) if (1..=proposals.len()).contains(&i) => proposals[i - 1].clone(),
            Ok(_) => {
                self.p.line(&format!(
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
            self.p.line(&format!("验收方式已确认：{command}"));
            self.start(Job::Task(prompt));
        }
    }

    async fn submit(&mut self, line: String) {
        if let Some(proposals) = self.proposals.take() {
            return self.answer_verify(&line, proposals);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
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
            self.p.line(&msg);
            return;
        }
        if name.is_empty()
            && let Some(sup) = self.sup()
            && let Some(reply) = crate::subagents::direct_message(&sup, &line)
        {
            self.p.line(&format!("➤ {reply}"));
            return;
        }
        if self.busy() {
            match name.as_str() {
                "" => {
                    self.steer
                        .push_item(line.clone(), vec![], STEER, "user", vec![]);
                    self.p.line(&format!("➤ 已排队插话：{}", first_line(&line)));
                }
                "queue" if !arg.is_empty() => {
                    self.steer
                        .push_item(arg.to_string(), vec![], FOLLOW_UP, "user", vec![]);
                    self.p.line(&format!(
                        "➤ 已排队，当前任务完成后处理：{}",
                        first_line(arg)
                    ));
                }
                "help" => self.p.line(HELP_TEXT),
                "tracing-viewer" => {
                    let msg = self.viewer.command(arg, Some(&self.reported_session));
                    self.p.line(&msg);
                }
                _ => self
                    .p
                    .line("当前任务进行中，该命令暂不可用（Esc/Ctrl+C 中断，或等待完成）"),
            }
            return;
        }
        if name.is_empty() {
            return self.start(Job::Task(line));
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
            "quit" | "exit" => self.quit = true,
            "help" => {
                let daily = agent.daily_mode();
                self.p.line(&format!(
                    "{HELP_TEXT}{}",
                    if daily { DAILY_HELP_TEXT } else { "" }
                ));
            }
            "queue" => {
                if arg.is_empty() {
                    self.p.line("用法：/queue <文本>");
                } else {
                    self.start(Job::Task(arg.to_string()));
                }
            }
            "new" => {
                agent.new_session();
                self.p.total_tokens = 0;
                self.p.line("已开始新会话");
            }
            "compact" => self.start(Job::Compact),
            "model" => {
                if arg.is_empty() {
                    let m = agent.client.model_id().to_string();
                    self.p.line(&format!("当前模型：{m}"));
                } else {
                    let (model, level) = crate::llm::catalog::split_model_arg(arg);
                    match agent.set_model(&model) {
                        Ok(()) => {
                            if level.is_some() {
                                agent.set_thinking(level);
                            }
                            let m = agent.client.model_id().to_string();
                            self.p.line(&format!("已切换到 {m}"));
                        }
                        Err(e) => self.p.line(&format!("✗ {e}")),
                    }
                }
            }
            "fork" => {
                if !arg.is_empty() && arg.parse::<usize>().is_err() {
                    return self
                        .p
                        .line("用法：/fork [n]（n = 保留前 n 轮对话，省略则全部保留）");
                }
                let parent = agent.session_id().to_string();
                let dir = agent.session_dir.clone();
                match agent.fork(arg.parse().ok()) {
                    Ok(path) => match Agent::resume(agent.config.clone(), &path, &dir) {
                        Ok(a) => {
                            self.steer = a.steer.clone();
                            let id = a.session_id().to_string();
                            self.agent = Some(a);
                            self.p.line(&format!("已从会话 {parent} 分叉，新会话 {id}"));
                        }
                        Err(e) => self.p.line(&format!("✗ {e}")),
                    },
                    Err(e) => self.p.line(&format!("✗ {e}")),
                }
            }
            "handoff" => {
                if arg.is_empty() {
                    return self.p.line("用法：/handoff <交接说明>");
                }
                let parent = agent.session_id().to_string();
                let prompt = agent.handoff_prompt(arg);
                let meta = agent.handoff_meta();
                agent.reset_to(meta);
                let id = agent.session_id().to_string();
                self.p
                    .line(&format!("已交接：新会话 {id}，关联原会话 {parent}"));
                self.start(Job::Task(prompt));
            }
            "plan" => match parse_toggle(arg, agent.plan_mode()) {
                None => self.p.line(cmds::PLAN_USAGE),
                Some(on) if agent.set_plan_mode(on) => self.p.line(cmds::plan_notice(on)),
                Some(on) => self.p.line(&cmds::unchanged_notice("计划模式", on)),
            },
            "code-mode" => match parse_toggle(arg, agent.code_mode()) {
                None => self.p.line(cmds::CODE_MODE_USAGE),
                Some(on) if agent.set_code_mode(on) => self.p.line(cmds::code_mode_notice(on)),
                Some(on) => self.p.line(&cmds::unchanged_notice("Code mode ", on)),
            },
            "daily-mode" => match parse_toggle(arg, agent.daily_mode()) {
                None => self.p.line(cmds::DAILY_USAGE),
                Some(on) if agent.set_daily_mode(on) => {
                    self.p.line(cmds::daily_toggle_notice(on));
                    if let Some(n) = cmds::web_search_missing_notice(agent) {
                        self.p.line(&n);
                    }
                }
                Some(on) => self.p.line(&cmds::unchanged_notice("Daily mode ", on)),
            },
            "memory" | "learn" if !agent.daily_mode() => {
                self.p.line(&cmds::daily_only_notice(&format!("/{name}")))
            }
            "memory" => match crate::memory::parse_memory_command(arg) {
                Err(()) => self.p.line(crate::memory::MEMORY_USAGE),
                Ok(target) => {
                    let store = agent.memory_store();
                    match (target, &store) {
                        (Some(t), Some(s)) => self.p.line(&format!(
                            "直接编辑 {}，从下一个会话起生效",
                            s.path(&t).display()
                        )),
                        _ => {
                            let text =
                                crate::memory::describe(store.as_ref(), agent.memory.as_ref());
                            self.p.line(&text);
                        }
                    }
                }
            },
            "learn" => {
                let msg = crate::learn::start_review(agent, false);
                self.p.line(&msg);
            }
            "btw" => {
                if arg.is_empty() {
                    return self.p.line("用法：/btw <问题>");
                }
                let mut rx = cmds::btw_query(&agent.client, &agent.messages, arg);
                self.p.line(&format!("💬 btw · {arg}"));
                while let Some(item) = rx.recv().await {
                    match item {
                        Ok(t) => {
                            print!("{t}");
                            let _ = std::io::stdout().flush();
                        }
                        Err(e) => {
                            println!();
                            self.p.line(&format!("✗ {e}"));
                        }
                    }
                }
                println!();
                self.p.line("（侧问答案不会进入会话）");
            }
            "goal" => match parse_goal_args(arg) {
                GoalCommand::Query => match pump::live_goal(agent) {
                    None => self.p.line("当前没有 goal"),
                    Some(g) => {
                        let t = cmds::goal_status_text(g);
                        self.p.line(&t);
                    }
                },
                GoalCommand::Clear => {
                    if pump::clear_goal(agent) {
                        self.p.line("已退出 goal 模式");
                    } else {
                        self.p.line("当前没有 goal");
                    }
                }
                GoalCommand::Set(text) => {
                    let prompt = pump::set_goal(agent, &text);
                    self.proposing = true;
                    self.start(Job::Task(prompt));
                }
            },
            "tracing-viewer" => {
                let id = agent.session_id().to_string();
                let msg = self.viewer.command(arg, Some(&id));
                self.p.line(&msg);
            }
            other => self
                .p
                .line(&format!("未知命令 /{other}（/help 查看可用命令）")),
        }
    }
}

impl Repl {
    fn on_background(&mut self, ev: BackgroundEvent) {
        if self.p.json {
            let v = match &ev {
                BackgroundEvent::Notice { kind, text } => {
                    json!({"type": "subagent_notice", "kind": kind, "text": text})
                }
                BackgroundEvent::TaskChanged { id, status, title } => {
                    json!({"type": "task", "id": id, "status": status, "title": title})
                }
                BackgroundEvent::PlanProposed(ids) => {
                    json!({"type": "plan_proposed", "task_ids": ids})
                }
                BackgroundEvent::MainWake => json!({"type": "main_wake"}),
                BackgroundEvent::AuthChecked { .. } => return,
            };
            println!("{v}");
        } else {
            match &ev {
                BackgroundEvent::Notice { kind, text } if !text.is_empty() => {
                    let icon = match kind.as_str() {
                        "result" => "✓",
                        "alert" => "⚠",
                        "question" => "?",
                        _ => "◆",
                    };
                    self.p.line(&format!("{icon} {text}"));
                }
                BackgroundEvent::PlanProposed(_) => {
                    if let Some(sup) = self.sup() {
                        self.p.line("◆ 子任务计划待确认：");
                        for l in sup.status_text(None).lines() {
                            self.p.line(&format!("  {l}"));
                        }
                        self.p
                            .line("  /task approve all 开始 · /task reject all 放弃");
                    }
                }
                _ => {}
            }
        }
        if matches!(ev, BackgroundEvent::MainWake) {
            self.maybe_wake_main();
        }
        self.finish_if_eof();
    }

    fn sup(&self) -> Option<std::sync::Arc<crate::subagents::Supervisor>> {
        self.agent
            .as_ref()
            .and_then(|a| a.supervisor.clone())
            .or_else(|| self.sup_handle.clone())
    }

    /// Start a main-agent turn for subagent notices queued while idle.
    fn maybe_wake_main(&mut self) {
        if self.busy() || self.proposals.is_some() || self.quit {
            return;
        }
        let Some(text) = self.sup().and_then(|s| s.take_wake_input()) else {
            return;
        };
        self.p.line("◆ 子代理有新进展，交给主 agent 处理");
        self.start(Job::Task(text));
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

pub async fn run_repl(agent: Agent, json_mode: bool) -> i32 {
    use std::io::IsTerminal;
    // OSC 7501 shares stdout with the transcript: never in --json mode or
    // into a pipe.
    let reporters = std::sync::Arc::new(Reporters::from_env(
        !json_mode && std::io::stdout().is_terminal(),
    ));
    crate::integrations::install_exit_hooks(reporters.clone());
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut repl = Repl {
        steer: agent.steer.clone(),
        viewer: crate::trace_viewer::Viewer::new(
            agent.session_dir.clone(),
            Some(agent.workdir.clone()),
        ),
        agent: Some(agent),
        job: None,
        cancel: CancellationToken::new(),
        tx,
        p: Printer {
            json: json_mode,
            in_text: false,
            total_tokens: 0,
        },
        proposing: false,
        proposals: None,
        quit: false,
        eof: false,
        eof_waiting: false,
        sup_handle: None,
        reporters: reporters.clone(),
        outcome: TurnOutcome::default(),
        reported_session: String::new(),
    };
    if let Some(a) = &repl.agent {
        note_session(&mut repl.reported_session, &repl.reporters, a);
    }
    let mut bg = crate::background::take_receiver();
    if let Some(a) = &repl.agent {
        repl.p.total_tokens = a.session_total_tokens();
        if !json_mode {
            println!(
                "limbo (headless) · {} · {}",
                a.client.model_id(),
                a.workdir.display()
            );
            println!(
                "session {} ({}) · /help 查看命令 · /quit 退出",
                a.session_id(),
                a.session_path().display()
            );
            if let Some(g) = pump::live_goal(a) {
                println!("🎯 goal：{}（{}）", g.text, g.status);
            }
            if let Some(n) = cmds::web_search_missing_notice(a) {
                println!("{n}");
            }
        }
    }
    let (line_tx, mut lines) = mpsc::unbounded_channel::<Option<String>>();
    tokio::spawn(async move {
        let mut input = BufReader::new(tokio::io::stdin()).lines();
        loop {
            match input.next_line().await {
                Ok(Some(l)) => {
                    if line_tx.send(Some(l)).is_err() {
                        return;
                    }
                }
                _ => {
                    let _ = line_tx.send(None);
                    return;
                }
            }
        }
    });
    while !repl.quit {
        tokio::select! {
            Some(ev) = rx.recv() => {
                repl.outcome.observe(&ev);
                repl.p.event(&ev);
            }
            done = async { repl.job.as_mut().unwrap().await }, if repl.job.is_some() => {
                // Flush events emitted just before the task returned.
                while let Ok(ev) = rx.try_recv() {
                    repl.outcome.observe(&ev);
                    repl.p.event(&ev);
                }
                match done {
                    Ok(agent) => repl.job_done(agent),
                    Err(e) => {
                        repl.p.line(&format!("✗ 内部错误：{e}"));
                        return 1;
                    }
                }
            }
            line = lines.recv(), if !repl.eof => match line.flatten() {
                Some(l) => repl.submit(l).await,
                None => {
                    repl.eof = true;
                    if repl.busy() {
                        repl.p.line("输入已关闭，等待当前任务完成…");
                    } else {
                        repl.finish_if_eof();
                    }
                }
            },
            Some(ev) = async { bg.as_mut().unwrap().recv().await }, if bg.is_some() => repl.on_background(ev),
            _ = tokio::signal::ctrl_c() => {
                if repl.proposals.take().is_some() {
                    repl.p.line(cmds::VERIFY_SKIPPED);
                } else if repl.busy() {
                    repl.cancel.cancel();
                } else {
                    repl.quit = true;
                }
            }
        }
    }
    if let Some(job) = repl.job.take() {
        repl.cancel.cancel();
        if let Ok(a) = job.await {
            repl.agent = Some(a);
        }
    }
    if let Some(a) = repl.agent.as_mut() {
        a.checkpoint();
        a.trace.flush();
    }
    repl.viewer.stop();
    if let Some(sup) = repl.sup() {
        sup.stop();
    }
    repl.reporters.release();
    0
}
