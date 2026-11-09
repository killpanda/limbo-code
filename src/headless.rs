//! Headless frontend: a line-oriented REPL on plain stdout (for scripts and
//! terminal multiplexers such as herdr), plus `--json` (one JSON object per
//! event) and `-p` one-shot mode.
//!
//! stdin is read concurrently with the running task, so a line typed while
//! the agent works is *steered* into the turn (`/queue <text>` waits for the
//! whole task instead). All orchestration — follow-ups, the goal loop, queue
//! policy — lives in [`crate::pump`]; this module only translates lines into
//! calls and events into text.

use crate::integrations::Reporters;
use std::io::Write;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEvent};
use crate::background::BackgroundEvent;
use crate::commands as cmds;
use crate::conversation::{Conversation, Sink};
use crate::pump;
use crate::subagents::describe_call;

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
            provider,
        } => {
            json!({"type": "tool_call", "id": id, "name": name, "arguments": arguments, "provider": provider})
        }
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
                name,
                arguments,
                provider,
                ..
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
                self.line(&format!("⚙ {desc} · {}", provider.label()));
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

impl Sink for Printer {
    fn line(&mut self, text: &str) {
        Printer::line(self, text);
    }
    fn btw_chunk(&mut self, text: &str) {
        print!("{text}");
        let _ = std::io::stdout().flush();
    }
    fn btw_end(&mut self) {
        println!();
    }
    fn idle(&mut self) {
        Printer::idle(self);
    }
    fn session_changed(&mut self, agent: &Agent) {
        self.total_tokens = agent.session_total_tokens();
    }
}

/// The stdin/stdout adapter around a [`Conversation`].
struct Repl {
    conv: Conversation<Printer>,
    eof: bool,
    eof_waiting: bool,
}

impl Repl {
    fn job_done(&mut self, agent: Agent) {
        self.conv.job_done(agent);
        self.finish_if_eof();
    }

    /// After stdin closed: quit once the main agent is idle and no subagent
    /// is still working (their results would otherwise be lost).
    fn finish_if_eof(&mut self) {
        let c = &mut self.conv;
        if !self.eof || c.busy() || c.quit {
            return;
        }
        if c.proposals.take().is_some() {
            c.sink.line("输入已关闭，跳过验收确认（保持单轮模式）");
        }
        // Results that already arrived go to the main agent first.
        c.maybe_wake_main();
        if c.busy() {
            return;
        }
        if c.sup().is_some_and(|s| s.working()) {
            if !std::mem::replace(&mut self.eof_waiting, true) {
                c.sink.line("输入已关闭，等待子代理完成…（Ctrl+C 退出）");
            }
            return;
        }
        c.quit = true;
    }

    fn on_background(&mut self, ev: BackgroundEvent) {
        if self.conv.sink.json {
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
            for l in self.conv.background_lines(&ev) {
                self.conv.sink.line(&l);
            }
        }
        self.conv.after_background(&ev);
        self.finish_if_eof();
    }
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
    let printer = Printer {
        json: json_mode,
        in_text: false,
        total_tokens: agent.session_total_tokens(),
    };
    let mut repl = Repl {
        conv: Conversation::new(agent, printer, tx, reporters.clone()),
        eof: false,
        eof_waiting: false,
    };
    let mut bg = crate::background::take_receiver();
    if let Some(a) = &repl.conv.agent
        && !json_mode
    {
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
    while !repl.conv.quit {
        tokio::select! {
            Some(ev) = rx.recv() => {
                repl.conv.observe(&ev);
                repl.conv.sink.event(&ev);
            }
            Some(done) = repl.conv.join(), if repl.conv.busy() => {
                // Flush events emitted just before the task returned.
                while let Ok(ev) = rx.try_recv() {
                    repl.conv.observe(&ev);
                    repl.conv.sink.event(&ev);
                }
                match done {
                    Ok(agent) => repl.job_done(agent),
                    Err(e) => {
                        repl.conv.sink.line(&format!("✗ 内部错误：{e}"));
                        return 1;
                    }
                }
            }
            line = lines.recv(), if !repl.eof => match line.flatten() {
                Some(l) => repl.conv.submit(l, vec![]).await,
                None => {
                    repl.eof = true;
                    if repl.conv.busy() {
                        repl.conv.sink.line("输入已关闭，等待当前任务完成…");
                    } else {
                        repl.finish_if_eof();
                    }
                }
            },
            Some(ev) = async { bg.as_mut().unwrap().recv().await }, if bg.is_some() => repl.on_background(ev),
            _ = tokio::signal::ctrl_c() => {
                if repl.conv.proposals.take().is_some() {
                    repl.conv.sink.line(cmds::VERIFY_SKIPPED);
                } else if repl.conv.busy() {
                    repl.conv.interrupt();
                } else {
                    repl.conv.quit = true;
                }
            }
        }
    }
    repl.conv.shutdown().await;
    reporters.release();
    0
}
