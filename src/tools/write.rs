//! `write`: create or overwrite a file.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolContext, arg_str, mutation_lock, resolve_path};
use crate::models::ToolResult;

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }
    fn effect(&self) -> crate::effect::Effect {
        crate::effect::Effect::Write
    }
    fn description(&self) -> String {
        "Create or overwrite a file. Use only for new files or complete rewrites.".into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        })
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let raw = arg_str(&args, "path").unwrap_or("");
        if raw.is_empty() {
            return ToolResult::err("path is required.");
        }
        let content = arg_str(&args, "content").unwrap_or("").to_string();
        let target = resolve_path(&ctx.workdir, raw);
        let lock = mutation_lock(&target);
        let _guard = lock.lock().await;
        if let Some(parent) = target.parent()
            && let Err(e) = tokio::fs::create_dir_all(parent).await
        {
            return ToolResult::err(format!("Could not write file: {e}"));
        }
        match tokio::fs::write(&target, content).await {
            Ok(()) => ToolResult::ok(format!("Wrote {raw}.")),
            Err(e) => ToolResult::err(format!("Could not write file: {e}")),
        }
    }
}
