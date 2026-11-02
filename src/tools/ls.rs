//! `ls`: list a directory (dirs first, dotfiles included).

use async_trait::async_trait;
use serde_json::{Value, json};

use super::truncate::truncate_head;
use super::{Tool, ToolContext, arg_i64, arg_str, resolve_path};
use crate::models::ToolResult;

const MAX_ENTRIES: usize = 500;

pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &'static str {
        "ls"
    }
    fn description(&self) -> String {
        format!(
            "List directory contents. Includes dotfiles. Returns up to {MAX_ENTRIES} entries by default (raise \
             `limit` for more; a notice is shown whenever entries are truncated)."
        )
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list"},
                "limit": {"type": "integer", "description": "Max entries"},
            },
            "required": [],
        })
    }
    fn replay_safe(&self) -> bool {
        true
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let raw = arg_str(&args, "path").unwrap_or(".");
        let limit = arg_i64(&args, "limit").map_or(MAX_ENTRIES, |l| l.max(1) as usize);
        let target = resolve_path(&ctx.workdir, raw);
        if !target.exists() {
            return ToolResult::err(format!("Path not found: {raw}"));
        }
        if !target.is_dir() {
            return ToolResult::err(format!("Not a directory: {raw}"));
        }
        let mut entries: Vec<(bool, String)> = match std::fs::read_dir(&target) {
            Ok(rd) => rd
                .flatten()
                .map(|e| {
                    let is_dir = e.path().is_dir();
                    (!is_dir, e.file_name().to_string_lossy().into_owned())
                })
                .collect(),
            Err(e) => return ToolResult::err(format!("Could not list directory: {e}")),
        };
        // (is_file, lowercase name): directories sort first.
        entries.sort_by_key(|(is_file, name)| (*is_file, name.to_lowercase()));
        if entries.is_empty() {
            return ToolResult::ok("(empty directory)");
        }
        let lines: Vec<String> = entries
            .iter()
            .take(limit)
            .map(|(is_file, n)| if *is_file { n.clone() } else { format!("{n}/") })
            .collect();
        let mut out = lines.join("\n");
        let mut notices = Vec::new();
        if entries.len() > limit {
            notices.push(format!(
                "Showing {limit} of {} entries. Use limit={} for more.",
                entries.len(),
                limit * 2
            ));
        }
        let t = truncate_head(&out, lines.len() + 1, super::truncate::DEFAULT_MAX_BYTES);
        if t.truncated {
            out = t.content;
            notices.push("Output truncated by size limit.".into());
        }
        if !notices.is_empty() {
            out.push_str(&format!("\n\n[{}]", notices.join(" ")));
        }
        ToolResult::ok(out)
    }
}
