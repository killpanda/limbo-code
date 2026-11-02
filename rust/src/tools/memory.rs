//! `memory` (daily mode): curate `USER.md` / `NOTES.md`.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolContext, arg_str};
use crate::memory::MemoryStore;
use crate::models::ToolResult;

const DESCRIPTION: &str = "Save durable facts to persistent memory that survives across sessions. Memory is \
injected into every future session, so keep entries compact and high-signal. Writes take effect from the next \
session.\n\nHOW: make all changes to one target in ONE call via 'operations' (each item: {action, content?, \
old_text?}). The batch applies atomically and the character limit is checked only on the final result, so one \
call can remove or patch stale entries to free room and add new ones.\n\nWHEN: only for facts that hold in every \
session regardless of task: who the user is, their preferences and style, stable environment facts, standing \
conventions, corrections the user made.\n\nSKIP: trivial or easily re-discovered facts, raw data, task progress, \
completed-work logs, temporary TODOs. Reusable procedures belong in a skill.\n\nTARGETS: 'user' = who the user is \
(name, role, preferences, style). 'notes' = your notes (environment, conventions, lessons).";

pub struct MemoryTool {
    pub store: MemoryStore,
}

#[async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &'static str {
        "memory"
    }
    fn description(&self) -> String {
        DESCRIPTION.into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["user", "notes"], "description": "'user' for the user profile, 'notes' for your notes."},
                "operations": {
                    "type": "array",
                    "description": "Changes to apply to the target, in order, all-or-nothing.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "patch", "remove"]},
                            "content": {"type": "string", "description": "add: the new entry. patch: the text that replaces old_text (the rest of the entry is kept)."},
                            "old_text": {"type": "string", "description": "patch/remove: text from exactly one entry. patch replaces this span; remove deletes the whole entry."},
                        },
                        "required": ["action"],
                    },
                },
            },
            "required": ["target", "operations"],
        })
    }
    async fn run(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        let Some(target) = arg_str(&args, "target") else {
            return ToolResult::err("target must be one of: user, notes");
        };
        let Some(ops) = args.get("operations").and_then(Value::as_array) else {
            return ToolResult::err("operations must be a list");
        };
        match self.store.apply(target, ops) {
            Ok(entries) => {
                let limit = self.store.limits.get(target).copied().unwrap_or(0);
                let used = MemoryStore::size(&entries);
                let cap = if limit > 0 {
                    format!("/{limit}")
                } else {
                    String::new()
                };
                let file = if target == "user" {
                    "USER.md"
                } else {
                    "NOTES.md"
                };
                ToolResult::ok(format!(
                    "Updated {file} ({} entries, {used}{cap} chars). Done; takes effect next session.",
                    entries.len()
                ))
            }
            Err(e) => ToolResult::err(e),
        }
    }
}
