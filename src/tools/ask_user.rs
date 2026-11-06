//! `ask_user` (daily mode): ask the user and end the turn. Their next
//! message is the answer, so the model never guesses one.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolContext, arg_str};
use crate::models::ToolResult;

pub const MAX_OPTIONS: usize = 6;

/// What the user sees: the question and its numbered options.
pub fn format_question(args: &Value) -> String {
    let mut lines = vec![format!(
        "❓ {}",
        arg_str(args, "question").unwrap_or("").trim()
    )];
    if let Some(opts) = args.get("options").and_then(Value::as_array) {
        for (i, o) in opts.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, o.as_str().unwrap_or("")));
        }
        if !opts.is_empty() {
            lines.push("  回复编号选择，或直接输入你的回答".into());
        }
    }
    lines.join("\n")
}

pub struct AskUserTool;

#[async_trait]
impl Tool for AskUserTool {
    fn name(&self) -> &'static str {
        "ask_user"
    }
    fn description(&self) -> String {
        "Ask the user a question when you need their decision or a fact only they know, instead of guessing. \
         Give up to 6 short options when the choice is between known alternatives (they may still answer freely). \
         Your turn ends after this call; their next message is the answer, and a bare number picks that option."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "One clear question."},
                "options": {"type": "array", "items": {"type": "string"}, "description": format!("Up to {MAX_OPTIONS} short options (optional).")},
            },
            "required": ["question"],
        })
    }
    fn effect(&self) -> crate::effect::Effect {
        crate::effect::Effect::Session
    }
    /// The answer comes from the user (their next message).
    fn provider(&self) -> crate::capability::Provider {
        crate::capability::Provider::Human
    }
    /// Asking again after a crash is harmless.
    fn recovery(&self) -> crate::effect::Recovery {
        crate::effect::Recovery::Replay
    }
    async fn run(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        if arg_str(&args, "question").is_none_or(|q| q.trim().is_empty()) {
            return ToolResult::err("question must be a non-empty string");
        }
        match args.get("options") {
            None | Some(Value::Null) => {}
            Some(Value::Array(a)) if a.iter().all(Value::is_string) => {
                if a.len() > MAX_OPTIONS {
                    return ToolResult::err(format!("at most {MAX_OPTIONS} options"));
                }
            }
            _ => return ToolResult::err("options must be a list of strings"),
        }
        let mut r = ToolResult::ok(format!(
            "{}\n\nAsked the user; end your turn now and wait for their reply.",
            format_question(&args)
        ));
        r.end_turn = true;
        r
    }
}
