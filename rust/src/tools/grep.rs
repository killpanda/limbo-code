//! `grep`: search file contents with ripgrep (a hard dependency, like the
//! Python version — one implementation of gitignore/glob semantics).

use async_trait::async_trait;
use serde_json::{Value, json};

use super::truncate::{MAX_OUTPUT_BYTES, truncate_output};
use super::{Tool, ToolContext, arg_bool, arg_i64, arg_str, display_path, resolve_path};
use crate::models::ToolResult;

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &'static str {
        "grep"
    }
    fn description(&self) -> String {
        "Search file contents for a pattern. Requires ripgrep (rg) on PATH. Respects .gitignore, glob filters, \
         and context lines."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "glob": {"type": "string", "description": "Filter files by glob (e.g. '*.py', 'src/**/*.ts')"},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive"},
                "fixed_string": {"type": "boolean", "description": "Literal string"},
                "context": {"type": "integer", "description": "Lines before/after match"},
                "limit": {"type": "integer", "description": "Max matches per file"},
            },
            "required": ["pattern"],
        })
    }
    fn replay_safe(&self) -> bool {
        true
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let pattern = arg_str(&args, "pattern").unwrap_or("");
        let raw_path = arg_str(&args, "path").unwrap_or(".");
        let target = resolve_path(&ctx.workdir, raw_path);
        if !target.exists() {
            return ToolResult::err(format!("Path not found: {raw_path}"));
        }
        let mut cmd = tokio::process::Command::new("rg");
        cmd.args([
            "--line-number",
            "--no-heading",
            "--color=never",
            "--no-require-git",
        ]);
        if arg_bool(&args, "ignore_case") {
            cmd.arg("--ignore-case");
        }
        if arg_bool(&args, "fixed_string") {
            cmd.arg("--fixed-strings");
        }
        if let Some(c) = arg_i64(&args, "context").filter(|c| *c > 0) {
            cmd.args(["-C", &c.to_string()]);
        }
        if let Some(g) = arg_str(&args, "glob").filter(|g| !g.is_empty()) {
            cmd.args(["-g", g]);
        }
        let limit = arg_i64(&args, "limit").unwrap_or(100).max(1);
        let shown = display_path(&ctx.workdir, &target);
        let shown = if shown.is_empty() {
            ".".to_string()
        } else {
            shown
        };
        cmd.args(["--max-count", &limit.to_string(), "--", pattern, &shown])
            .current_dir(&ctx.workdir);
        let out = match tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output()).await
        {
            Err(_) => return ToolResult::err("grep timed out."),
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                return ToolResult::err(
                    "ripgrep (rg) is required for grep but was not found on PATH. Install it (e.g. 'brew install \
                     ripgrep' or 'apt install ripgrep') and try again.",
                );
            }
            Ok(Err(e)) => return ToolResult::err(format!("grep failed: {e}")),
            Ok(Ok(o)) => o,
        };
        let text = truncate_output(
            String::from_utf8_lossy(&out.stdout).into_owned(),
            MAX_OUTPUT_BYTES,
        );
        if text.is_empty() {
            let err = String::from_utf8_lossy(&out.stderr);
            if out.status.code() == Some(2) && !err.trim().is_empty() {
                return ToolResult::err(format!("grep error: {}", err.trim()));
            }
            return ToolResult::ok("No matches.");
        }
        ToolResult::ok(text)
    }
}
