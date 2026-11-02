//! Subagents: the `task` tool runs a child agent with a fresh context.
//!
//! This is a deliberately simpler design than the Python event-sourced task
//! board: the main agent calls `task` (several calls in one response run in
//! parallel), each child works in its own session file under
//! `<parent id>.agents/`, and its final answer becomes the tool result.
//! Children cannot spawn children (depth 1).
//!
//! 🦀 This creates a recursive async call (Agent::run → task tool →
//! Agent::run). Plain recursive `async fn`s do not compile because their
//! future would have infinite size; the boxed future that `#[async_trait]`
//! produces breaks the cycle.

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::agent::{Agent, AgentEvent};
use crate::config::Config;
use crate::llm::LlmClient;
use crate::models::{Role, ToolResult};
use crate::tools::{Tool, ToolContext, ToolRegistry, arg_str};

pub struct TaskTool {
    config: Config,
}

impl TaskTool {
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

/// One-line summary of a tool call for progress display.
pub fn describe_call(name: &str, args: &Value) -> String {
    let key = match name {
        "read" | "write" | "edit" | "ls" => "path",
        "bash" => "command",
        "grep" | "find" => "pattern",
        "task" => "description",
        _ => "",
    };
    let detail = args.get(key).and_then(Value::as_str).unwrap_or("");
    let detail: String = detail
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(80)
        .collect();
    if detail.is_empty() {
        name.to_string()
    } else {
        format!("{name} {detail}")
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &'static str {
        "task"
    }
    fn description(&self) -> String {
        "Delegate a self-contained task to a subagent that starts with a fresh context and reports back a \
         summary. Use mode 'explore' (read-only: read/grep/find/ls) for research and code search, 'general' \
         for work that edits files or runs commands. Issue several task calls in one response to run them in \
         parallel. The prompt must contain everything the subagent needs: it cannot see this conversation."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "A short (3-6 word) label for the task"},
                "prompt": {"type": "string", "description": "Complete instructions for the subagent"},
                "mode": {"type": "string", "enum": ["explore", "general"], "description": "explore = read-only (default), general = full tools"},
            },
            "required": ["description", "prompt"],
        })
    }
    fn mutates(&self) -> bool {
        false // plan mode forces explore mode below instead of blocking
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let prompt = arg_str(&args, "prompt").unwrap_or("").to_string();
        if prompt.trim().is_empty() {
            return ToolResult::err("prompt is required.");
        }
        let label = arg_str(&args, "description")
            .unwrap_or("subagent")
            .to_string();
        let general = arg_str(&args, "mode") == Some("general") && !ctx.plan_mode;

        let mut cfg = self.config.clone();
        if !cfg.subagents.cheap_model.is_empty() {
            cfg.llm.model = cfg.subagents.cheap_model.clone();
        }
        cfg.llm.max_iterations = cfg.subagents.max_iterations;
        cfg.subagents.enabled = false; // depth 1
        let client = match LlmClient::new(&cfg) {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("Could not start subagent: {e}")),
        };
        let registry = if general {
            ToolRegistry::core(cfg.tools.bash_enabled)
        } else {
            ToolRegistry::read_only()
        };

        let parent = ctx
            .session_path
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("limbo-subagents.jsonl"));
        let parent_id = parent
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let dir = parent.with_file_name(format!("{parent_id}.agents"));
        let child_path = dir.join(format!("{}.jsonl", sanitize(&ctx.call_id)));
        let mut child = Agent::new_subagent(
            cfg,
            client,
            registry,
            &ctx.workdir,
            child_path.clone(),
            &parent_id,
            &ctx.call_id,
        );
        child.meta.title = label.clone();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let progress_ctx = ctx.clone();
        // Forward the child's activity as one-line progress for the UI.
        let forward = tokio::spawn(async move {
            let mut tools = 0usize;
            while let Some(ev) = rx.recv().await {
                match ev {
                    AgentEvent::ToolCall {
                        name, arguments, ..
                    } => {
                        tools += 1;
                        progress_ctx.progress(format!("→ {}", describe_call(&name, &arguments)));
                    }
                    AgentEvent::Retry { attempt, .. } => {
                        progress_ctx.progress(format!("↻ retry {attempt}"))
                    }
                    AgentEvent::Error(e) => progress_ctx.progress(format!("✗ {e}")),
                    _ => {}
                }
            }
            tools
        });
        child.run(prompt, vec![], &tx, &ctx.cancel).await;
        drop(tx);
        let tool_count = forward.await.unwrap_or(0);

        let answer = child
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant && !m.text().trim().is_empty())
            .map(|m| m.text().to_string())
            .unwrap_or_else(|| "(the subagent produced no answer)".into());
        let usage = child.session_total_tokens();
        let header = format!(
            "[subagent '{label}' finished: {tool_count} tool calls, {usage} tokens; transcript: {}]",
            child_path.display()
        );
        if ctx.cancel.is_cancelled() {
            return ToolResult {
                success: false,
                output: Some(format!("{header}\n(interrupted)\n{answer}")),
                error: Some("interrupted".into()),
                attachments: None,
            };
        }
        ToolResult::ok(format!("{header}\n\n{answer}"))
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::agent::tests::{fake_server, sse, test_config};
    use crate::agent::{Agent, AgentEvent};
    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn task_tool_runs_a_child_agent() {
        let dir = tempfile::tempdir().unwrap();
        let parent1 = sse(&[
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"task","arguments":"{\"description\":\"scan\",\"prompt\":\"list files\"}"}}]},"finish_reason":"tool_calls"}]}),
        ]);
        let child1 = sse(&[
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"k1","function":{"name":"ls","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}),
        ]);
        let child2 = sse(&[
            json!({"choices":[{"delta":{"content":"There are 0 files."},"finish_reason":"stop"}]}),
        ]);
        let parent2 =
            sse(&[json!({"choices":[{"delta":{"content":"Done."},"finish_reason":"stop"}]})]);
        let (base, seen) = fake_server(vec![parent1, child1, child2, parent2]).await;
        let sessions = dir.path().join("sessions");
        let mut agent = Agent::new(test_config(&base), dir.path(), &sessions).unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        agent
            .run("scan please".into(), vec![], &tx, &CancellationToken::new())
            .await;
        drop(tx);
        let mut progress = vec![];
        let mut result = String::new();
        while let Some(e) = rx.recv().await {
            match e {
                AgentEvent::ToolProgress { line, .. } => progress.push(line),
                AgentEvent::ToolResult { output, .. } => result = output,
                _ => {}
            }
        }
        assert_eq!(progress, vec!["→ ls"]);
        assert!(result.contains("There are 0 files."), "{result}");
        assert_eq!(agent.messages.last().unwrap().text(), "Done.");
        // The child only had read-only tools, and no task tool (depth 1).
        let bodies = seen.lock().unwrap().clone();
        let child_tools: Vec<&str> = bodies[1]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(child_tools, vec!["read", "grep", "find", "ls"]);
        // Its transcript lives next to the parent session.
        let agents_dir = sessions.join(format!("{}.agents", agent.session_id()));
        assert!(agents_dir.join("t1.jsonl").exists());
    }
}
