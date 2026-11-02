//! Headless frontend: a line-oriented REPL on plain stdout (for scripts and
//! terminal multiplexers), plus `--json` (one JSON object per event) and
//! `-p` one-shot mode.

use std::io::Write;

use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEvent};
use crate::subagent::describe_call;

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
        AgentEvent::Warning(t) => json!({"type": "warning", "message": t}),
        AgentEvent::Error(t) => json!({"type": "error", "message": t}),
        AgentEvent::Interrupted => json!({"type": "interrupted"}),
        AgentEvent::TurnEnd => json!({"type": "turn_end"}),
    }
}

fn print_event(ev: &AgentEvent, json_mode: bool, in_text: &mut bool) {
    let mut out = std::io::stdout();
    if json_mode {
        let _ = writeln!(out, "{}", event_json(ev));
        return;
    }
    let mut end_text = |out: &mut std::io::Stdout| {
        if *in_text {
            let _ = writeln!(out);
            *in_text = false;
        }
    };
    match ev {
        AgentEvent::Text(t) => {
            *in_text = true;
            let _ = write!(out, "{t}");
        }
        AgentEvent::Thinking(_) | AgentEvent::TurnStart | AgentEvent::Usage { .. } => {}
        AgentEvent::ToolCall {
            name, arguments, ..
        } => {
            end_text(&mut out);
            let _ = writeln!(out, "⚙ {}", describe_call(name, arguments));
        }
        AgentEvent::ToolProgress { line, .. } => {
            let _ = writeln!(out, "  │ {line}");
        }
        AgentEvent::ToolResult {
            success, output, ..
        } => {
            let first = output.lines().next().unwrap_or("");
            let _ = writeln!(
                out,
                "  {} {}",
                if *success { "✓" } else { "✗" },
                crate::llm::truncate_chars(first, 120)
            );
        }
        AgentEvent::Retry { attempt, reason } => {
            end_text(&mut out);
            let _ = writeln!(out, "↻ retry {attempt}: {reason}");
        }
        AgentEvent::Compaction {
            compacted,
            reason,
            tokens_before,
            tokens_after,
            ..
        } => {
            end_text(&mut out);
            if *compacted {
                let _ = writeln!(
                    out,
                    "⊙ compacted context {tokens_before} → {tokens_after} tokens"
                );
            } else {
                let _ = writeln!(out, "⊙ {}", reason.clone().unwrap_or_default());
            }
        }
        AgentEvent::Steer(t) => {
            end_text(&mut out);
            let _ = writeln!(out, "❯ (queued) {t}");
        }
        AgentEvent::Warning(t) => {
            end_text(&mut out);
            let _ = writeln!(out, "⚠ {t}");
        }
        AgentEvent::Error(t) => {
            end_text(&mut out);
            let _ = writeln!(out, "✗ {t}");
        }
        AgentEvent::Interrupted => {
            end_text(&mut out);
            let _ = writeln!(out, "⏹ interrupted");
        }
        AgentEvent::TurnEnd => end_text(&mut out),
    }
    let _ = out.flush();
}

async fn run_turn(agent: &mut Agent, input: String, json_mode: bool) {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let c2 = cancel.clone();
    // Ctrl+C interrupts the turn instead of killing the process.
    let ctrlc = tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            c2.cancel();
        }
    });
    let printer = tokio::spawn(async move {
        let mut in_text = false;
        while let Some(ev) = rx.recv().await {
            print_event(&ev, json_mode, &mut in_text);
        }
    });
    agent.run(input, vec![], &tx, &cancel).await;
    drop(tx);
    let _ = printer.await;
    ctrlc.abort();
}

/// One prompt, print the answer, exit.
pub async fn run_once(mut agent: Agent, prompt: String, json_mode: bool) -> i32 {
    run_turn(&mut agent, prompt, json_mode).await;
    0
}

pub async fn run_repl(mut agent: Agent, json_mode: bool) -> i32 {
    if !json_mode {
        println!(
            "limbo (headless) · {} · {}",
            agent.client.model_id(),
            agent.workdir.display()
        );
        println!(
            "session {} ({}) · /quit to exit",
            agent.session_id(),
            agent.session_path().display()
        );
    }
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        if !json_mode {
            print!("❯ ");
            let _ = std::io::stdout().flush();
        }
        let Ok(Some(line)) = lines.next_line().await else {
            break;
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        match line.as_str() {
            "/quit" | "/exit" => break,
            "/new" => {
                agent.new_session();
                println!("new session {}", agent.session_id());
                continue;
            }
            "/compact" => {
                let (tx, mut rx) = mpsc::unbounded_channel();
                agent
                    .compact("manual", &tx, &CancellationToken::new())
                    .await;
                drop(tx);
                let mut t = false;
                while let Some(ev) = rx.recv().await {
                    print_event(&ev, json_mode, &mut t);
                }
                continue;
            }
            "/plan on" | "/plan" => {
                agent.set_plan_mode(true);
                println!("plan mode on");
                continue;
            }
            "/plan off" => {
                agent.set_plan_mode(false);
                println!("plan mode off");
                continue;
            }
            _ => {}
        }
        if let Some(model) = line.strip_prefix("/model ") {
            match agent.set_model(model.trim()) {
                Ok(()) => println!("model → {}", agent.client.model_id()),
                Err(e) => println!("✗ {e}"),
            }
            continue;
        }
        run_turn(&mut agent, line, json_mode).await;
    }
    agent.checkpoint();
    0
}
