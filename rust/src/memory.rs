//! Persistent curated memory (daily mode): `~/.limbo/memory/USER.md` and
//! `NOTES.md`, entries separated by `§`, each file with a character cap.
//!
//! A session takes a frozen [`MemorySnapshot`] when its agent is built: the
//! `memory` tool writes to disk at once, but the system prompt keeps the
//! snapshot so the prompt prefix (and its cache) never changes mid-session.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::config::limbo_home;

pub const ENTRY_DELIMITER: &str = "\n§\n";
pub const TARGETS: [(&str, &str); 2] = [("user", "USER.md"), ("notes", "NOTES.md")];
pub const ACTIONS: [&str; 3] = ["add", "patch", "remove"];

/// Appended to a daily user message after N turns without a memory write.
pub const MEMORY_NUDGE: &str = "\n\n<system-reminder>If this conversation revealed durable facts about the user \
or how they like things done, save them with the memory tool (skip it if nothing qualifies). Do not mention this \
reminder.</system-reminder>";

pub const MEMORY_USAGE: &str = "用法：/memory 查看记忆 · /memory edit [user|notes] 用编辑器打开";

pub fn memory_dir() -> PathBuf {
    limbo_home().join("memory")
}

fn file_for(target: &str) -> Option<&'static str> {
    TARGETS.iter().find(|(t, _)| *t == target).map(|(_, f)| *f)
}

fn xml_escape(t: &str) -> String {
    t.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemorySnapshot {
    pub user: Vec<String>,
    pub notes: Vec<String>,
}

impl MemorySnapshot {
    /// The `<memory>` prompt block, or None when empty.
    pub fn render(&self) -> Option<String> {
        let mut sections = Vec::new();
        if !self.user.is_empty() {
            let items: Vec<String> = self
                .user
                .iter()
                .map(|e| format!("- {}", xml_escape(e)))
                .collect();
            sections.push(format!(
                "<user_profile>\n{}\n</user_profile>",
                items.join("\n")
            ));
        }
        if !self.notes.is_empty() {
            let items: Vec<String> = self
                .notes
                .iter()
                .map(|e| format!("- {}", xml_escape(e)))
                .collect();
            sections.push(format!("<agent_notes>\n{}\n</agent_notes>", items.join("\n")));
        }
        if sections.is_empty() {
            return None;
        }
        Some(format!(
            "\n\n<memory>\nPersistent memory from earlier sessions (a snapshot taken when this session started; \
             facts the user states now take precedence):\n\n{}\n</memory>\n",
            sections.join("\n\n")
        ))
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStore {
    pub directory: PathBuf,
    pub limits: HashMap<String, usize>,
}

impl MemoryStore {
    pub fn new(directory: PathBuf, user_chars: usize, notes_chars: usize) -> Self {
        let limits = [
            ("user".to_string(), user_chars),
            ("notes".to_string(), notes_chars),
        ]
        .into();
        Self { directory, limits }
    }

    pub fn path(&self, target: &str) -> PathBuf {
        self.directory.join(file_for(target).unwrap_or("NOTES.md"))
    }

    pub fn entries(&self, target: &str) -> Vec<String> {
        let Ok(text) = std::fs::read_to_string(self.path(target)) else {
            return Vec::new();
        };
        text.split(ENTRY_DELIMITER.trim())
            .map(str::trim)
            .filter(|e| !e.is_empty())
            .map(String::from)
            .collect()
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        MemorySnapshot {
            user: self.entries("user"),
            notes: self.entries("notes"),
        }
    }

    pub fn size(entries: &[String]) -> usize {
        entries.join(ENTRY_DELIMITER).chars().count()
    }

    /// Apply `operations` all-or-nothing; the cap is checked on the result.
    pub fn apply(&self, target: &str, operations: &[Value]) -> Result<Vec<String>, String> {
        let Some(file) = file_for(target) else {
            return Err(format!("Unknown target {target:?}; use one of: user, notes"));
        };
        if operations.is_empty() {
            return Err("No operations given".into());
        }
        let path = self.path(target);
        let mut entries = self.entries(target);
        for (i, op) in operations.iter().enumerate() {
            let idx = (operations.len() > 1).then_some(i + 1);
            entries = apply_one(entries, op, idx)?;
        }
        let limit = self.limits.get(target).copied().unwrap_or(0);
        let size = Self::size(&entries);
        if limit > 0 && size > limit {
            return Err(format!(
                "{file} would be {size} characters; the limit is {limit}. Nothing was written. Reissue one batch \
                 that removes or shortens stale entries and adds the new one together. Current entries:\n{}",
                numbered(&self.entries(target))
            ));
        }
        write_entries(&path, &entries).map_err(|e| format!("Could not write memory: {e}"))?;
        Ok(entries)
    }
}

fn write_entries(path: &Path, entries: &[String]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if path.exists() {
        let _ = std::fs::copy(path, path.with_extension("md.bak"));
    }
    let mut text = entries.join(ENTRY_DELIMITER);
    if !entries.is_empty() {
        text.push('\n');
    }
    crate::session::atomic_write(path, text.as_bytes())
}

fn numbered(entries: &[String]) -> String {
    if entries.is_empty() {
        return "(empty)".into();
    }
    entries
        .iter()
        .enumerate()
        .map(|(i, e)| format!("{}. {e}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

fn apply_one(mut entries: Vec<String>, op: &Value, index: Option<usize>) -> Result<Vec<String>, String> {
    let w = index.map(|i| format!("operation {i}: ")).unwrap_or_default();
    let Some(op) = op.as_object() else {
        return Err(format!("{w}each operation must be an object"));
    };
    let action = op.get("action").and_then(Value::as_str).unwrap_or("");
    let content = op.get("content").and_then(Value::as_str);
    let old_text = op.get("old_text").and_then(Value::as_str);
    if !ACTIONS.contains(&action) {
        return Err(format!("{w}action must be one of: add, patch, remove"));
    }
    if action == "add" || action == "patch" {
        match content {
            Some(c) if !c.trim().is_empty() => {
                if c.contains(ENTRY_DELIMITER.trim()) {
                    return Err(format!(
                        "{w}content must not contain the entry separator '§'"
                    ));
                }
            }
            _ => return Err(format!("{w}{action} needs a non-empty content")),
        }
    }
    if action == "add" {
        let entry = content.unwrap().trim().to_string();
        if !entries.contains(&entry) {
            entries.push(entry);
        }
        return Ok(entries);
    }
    let Some(old) = old_text.filter(|o| !o.trim().is_empty()) else {
        return Err(format!("{w}{action} needs old_text (text from one entry)"));
    };
    let matches: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.contains(old))
        .map(|(i, _)| i)
        .collect();
    match matches.len() {
        0 => return Err(format!("{w}no entry contains {old:?}")),
        1 => {}
        n => return Err(format!("{w}{n} entries contain {old:?}; quote more of the entry")),
    }
    let i = matches[0];
    if action == "remove" {
        entries.remove(i);
        return Ok(entries);
    }
    if entries[i].matches(old).count() > 1 {
        return Err(format!(
            "{w}{old:?} occurs more than once in the entry; quote more of it"
        ));
    }
    entries[i] = entries[i].replace(old, content.unwrap()).trim().to_string();
    Ok(entries)
}

/// What `/memory` prints.
pub fn describe(store: Option<&MemoryStore>, snapshot: Option<&MemorySnapshot>) -> String {
    let Some(store) = store else {
        return "记忆已关闭（[daily] memory = false）".into();
    };
    let mut blocks = Vec::new();
    for (target, label) in [("user", "关于你"), ("notes", "笔记")] {
        let entries = store.entries(target);
        let limit = store.limits.get(target).copied().unwrap_or(0);
        let head = format!(
            "{label} · {} · {}/{limit} 字符",
            store.path(target).display(),
            MemoryStore::size(&entries)
        );
        let body = if entries.is_empty() {
            "  （空）".to_string()
        } else {
            entries
                .iter()
                .map(|e| format!("  - {e}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        blocks.push(format!("{head}\n{body}"));
    }
    if let Some(s) = snapshot
        && *s != store.snapshot()
    {
        blocks.push("本会话之后的改动从下一个会话起生效。".into());
    }
    blocks.join("\n\n")
}

/// `/memory` argument → `Ok(None)` (show), `Ok(Some(target))` (edit) or Err.
pub fn parse_memory_command(arg: &str) -> Result<Option<String>, ()> {
    let words: Vec<&str> = arg.split_whitespace().collect();
    match words.as_slice() {
        [] => Ok(None),
        ["edit"] => Ok(Some("user".into())),
        ["edit", t] if file_for(t).is_some() => Ok(Some(t.to_string())),
        _ => Err(()),
    }
}

/// `$VISUAL` / `$EDITOR` (default vi) as argv.
pub fn editor_command(path: &Path) -> Vec<String> {
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".into());
    let mut argv: Vec<String> = editor.split_whitespace().map(String::from).collect();
    argv.push(path.display().to_string());
    argv
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn add_patch_remove_and_cap() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path().into(), 40, 100);
        store
            .apply("user", &[json!({"action": "add", "content": "Name: Ada"})])
            .unwrap();
        store
            .apply(
                "user",
                &[json!({"action": "patch", "old_text": "Ada", "content": "Ada L."})],
            )
            .unwrap();
        assert_eq!(store.entries("user"), vec!["Name: Ada L."]);
        let err = store
            .apply(
                "user",
                &[json!({"action": "add", "content": "x".repeat(50)})],
            )
            .unwrap_err();
        assert!(err.contains("limit is 40"));
        // one batch frees room and adds
        store
            .apply(
                "user",
                &[
                    json!({"action": "remove", "old_text": "Ada"}),
                    json!({"action": "add", "content": "likes tea"}),
                ],
            )
            .unwrap();
        assert_eq!(store.entries("user"), vec!["likes tea"]);
        let snap = store.snapshot().render().unwrap();
        assert!(snap.contains("<user_profile>"));
    }
}
