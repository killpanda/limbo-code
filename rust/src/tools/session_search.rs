//! `session_search` (daily mode): recall past conversations by scanning
//! the session JSONL files — no index, no LLM call. Subagent sessions live
//! in `<id>.agents/` subdirectories and are never searched.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::truncate::{MAX_OUTPUT_BYTES, truncate_output};
use super::{Tool, ToolContext, arg_bool, arg_i64, arg_str};
use crate::models::ToolResult;

const MAX_FILES: usize = 500;
const MAX_HITS: usize = 10;
const SNIPPET_CHARS: usize = 300;
const READ_CHARS: usize = 2000;
const WINDOW: usize = 4;

struct Session {
    id: String,
    title: String,
    date: String,
    daily: bool,
    messages: Vec<(String, String)>, // (role, content)
}

fn load(path: &Path) -> Option<Session> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut meta: Option<Map<String, Value>> = None;
    let mut messages = Vec::new();
    for line in text.lines() {
        let Ok(Value::Object(rec)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match rec.get("type").and_then(Value::as_str) {
            Some("meta") => meta = Some(rec),
            Some("meta_patch") => {
                if let Some(m) = meta.as_mut() {
                    for (k, v) in rec {
                        if k != "type" {
                            m.insert(k, v);
                        }
                    }
                }
            }
            None => {
                let role = rec.get("role").and_then(Value::as_str).unwrap_or("");
                let content = rec.get("content").and_then(Value::as_str).unwrap_or("");
                if (role == "user" || role == "assistant") && !content.is_empty() {
                    messages.push((role.to_string(), content.to_string()));
                }
            }
            _ => {}
        }
    }
    let meta = meta?;
    let s = |k: &str| meta.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    Some(Session {
        id: Some(s("id")).filter(|i| !i.is_empty()).unwrap_or_else(|| {
            path.file_stem()
                .map(|x| x.to_string_lossy().into_owned())
                .unwrap_or_default()
        }),
        title: s("title"),
        date: s("created_at").chars().take(10).collect(),
        daily: meta.get("daily_mode").and_then(Value::as_bool).unwrap_or(false),
        messages,
    })
}

fn snippet(text: &str, terms: &[String]) -> String {
    let low = text.to_lowercase();
    let at_byte = terms.iter().filter_map(|t| low.find(t.as_str())).min().unwrap_or(0);
    // lowercase may shift byte offsets; map via char count of the prefix
    let at = low[..at_byte.min(low.len())].chars().count();
    let chars: Vec<char> = text.chars().collect();
    let start = at.saturating_sub(SNIPPET_CHARS / 3).min(chars.len());
    let end = (start + SNIPPET_CHARS).min(chars.len());
    let piece: String = chars[start..end].iter().collect();
    let piece = piece.split_whitespace().collect::<Vec<_>>().join(" ");
    format!(
        "{}{piece}{}",
        if start > 0 { "…" } else { "" },
        if end < chars.len() { "…" } else { "" }
    )
}

pub struct SessionSearchTool {
    pub dir: PathBuf,
}

impl SessionSearchTool {
    fn files(&self) -> Vec<PathBuf> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
            .map(|p| {
                let m = std::fs::metadata(&p)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::UNIX_EPOCH);
                (m, p)
            })
            .collect();
        files.sort_by(|a, b| b.0.cmp(&a.0));
        files.into_iter().take(MAX_FILES).map(|(_, p)| p).collect()
    }

    fn read(&self, reference: &str, around: Option<i64>) -> ToolResult {
        let files = self.files();
        let stem = |p: &PathBuf| {
            p.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let matches: Vec<&PathBuf> = files.iter().filter(|p| stem(p).starts_with(reference)).collect();
        if matches.is_empty() {
            return ToolResult::err(format!("no session matches {reference:?}"));
        }
        let exact = matches.iter().find(|p| stem(p) == reference);
        if matches.len() > 1 && exact.is_none() {
            return ToolResult::err(format!(
                "{} sessions match {reference:?}; give more of the id",
                matches.len()
            ));
        }
        let path = exact.copied().unwrap_or(matches[0]);
        let Some(session) = load(path) else {
            return ToolResult::err(format!("cannot read session {reference:?}"));
        };
        let all: Vec<(i64, (String, String))> = session
            .messages
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, m)| (i as i64, m))
            .collect();
        let shown: Vec<(i64, (String, String))> = if let Some(a) = around {
            let lo = (a - WINDOW as i64).max(0) as usize;
            let hi = ((a + WINDOW as i64 + 1).max(0) as usize).min(all.len());
            all.get(lo..hi).map(<[_]>::to_vec).unwrap_or_default()
        } else if all.len() > 2 * WINDOW + 2 {
            let head = &all[..WINDOW + 1];
            let tail = &all[all.len() - (WINDOW + 1)..];
            let skipped = all.len() - head.len() - tail.len();
            let mut v = head.to_vec();
            v.push((-1, ("…".into(), format!("{skipped} messages skipped"))));
            v.extend_from_slice(tail);
            v
        } else {
            all
        };
        let mut lines = vec![format!(
            "session {} · {} · {}",
            session.id,
            session.date,
            if session.title.is_empty() { "(untitled)" } else { &session.title }
        )];
        for (i, (role, content)) in shown {
            let mut text: String = content.chars().take(READ_CHARS).collect();
            if content.chars().count() > READ_CHARS {
                text.push('…');
            }
            let label = if i >= 0 { format!("[{i}] ") } else { String::new() };
            lines.push(format!("{label}{role}: {text}"));
        }
        ToolResult::ok(truncate_output(lines.join("\n\n"), MAX_OUTPUT_BYTES))
    }
}

#[async_trait]
impl Tool for SessionSearchTool {
    fn name(&self) -> &'static str {
        "session_search"
    }
    fn description(&self) -> String {
        "Search and reread past conversations with the user. Use it when the user refers to earlier work (\"what \
         did we decide about X\", \"continue the report from last week\") or a past session likely holds the \
         answer. With query: sessions whose messages contain every word, newest first, with snippets and message \
         indices. With session_id: read that session (around=<index> for the messages near one hit). Searches \
         daily sessions unless include_coding is true."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Words that must all appear."},
                "session_id": {"type": "string", "description": "A session id (or prefix) to read."},
                "around": {"type": "integer", "description": "With session_id: a message index."},
                "include_coding": {"type": "boolean", "description": "Also search coding sessions (default false)."},
            },
        })
    }
    fn replay_safe(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        if let Some(id) = arg_str(&args, "session_id").map(str::trim).filter(|s| !s.is_empty()) {
            return self.read(id, arg_i64(&args, "around"));
        }
        let Some(query) = arg_str(&args, "query").map(str::trim).filter(|q| !q.is_empty()) else {
            return ToolResult::err("give query (words to find) or session_id (a session to read)");
        };
        let terms: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        let include_coding = arg_bool(&args, "include_coding");
        let mut blocks = Vec::new();
        for path in self.files() {
            let Some(session) = load(&path) else { continue };
            if !(session.daily || include_coding) {
                continue;
            }
            let hits: Vec<(usize, &(String, String))> = session
                .messages
                .iter()
                .enumerate()
                .filter(|(_, (_, c))| {
                    let low = c.to_lowercase();
                    terms.iter().all(|t| low.contains(t.as_str()))
                })
                .collect();
            if hits.is_empty() {
                continue;
            }
            let mut lines = vec![format!(
                "session {} · {} · {}{}",
                session.id,
                session.date,
                if session.title.is_empty() { "(untitled)" } else { &session.title },
                if session.daily { "" } else { " · coding" }
            )];
            for (i, (role, content)) in hits.iter().take(3) {
                lines.push(format!("  [{i}] {role}: {}", snippet(content, &terms)));
            }
            if hits.len() > 3 {
                lines.push(format!("  … {} more matching messages", hits.len() - 3));
            }
            blocks.push(lines.join("\n"));
            if blocks.len() >= MAX_HITS {
                break;
            }
        }
        if blocks.is_empty() {
            return ToolResult::ok(format!("No past sessions mention: {query}"));
        }
        ToolResult::ok(truncate_output(
            format!(
                "{}\n\nRead one with session_id (and around=<index> for context).",
                blocks.join("\n\n")
            ),
            MAX_OUTPUT_BYTES,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn finds_and_reads() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("abc123.jsonl"),
            "{\"type\":\"meta\",\"id\":\"abc123\",\"title\":\"Trip\",\"created_at\":\"2026-01-02T00:00:00\",\"daily_mode\":true}\n\
             {\"role\":\"user\",\"content\":\"plan a trip to Kyoto in spring\"}\n\
             {\"role\":\"assistant\",\"content\":\"Sure, Kyoto in April is lovely\"}\n",
        )
        .unwrap();
        let t = SessionSearchTool { dir: dir.path().into() };
        let ctx = ToolContext::simple(dir.path());
        let r = t.run(json!({"query": "kyoto SPRING"}), &ctx).await;
        let out = r.output.unwrap();
        assert!(out.contains("session abc123"), "{out}");
        assert!(out.contains("[0] user"));
        let r = t.run(json!({"session_id": "abc"}), &ctx).await;
        assert!(r.output.unwrap().contains("[1] assistant: Sure"));
    }
}
