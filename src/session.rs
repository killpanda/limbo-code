//! Session storage: the JSONL journal, byte-compatible with the Python version.
//!
//! Layout of `~/.limbo/sessions/<id>.jsonl`:
//! - line 1: `{"type": "meta", ...}`
//! - then, in the order they happened: message lines (no `type` key),
//!   `compaction`, `tool_intent`, `meta_patch`, `queue` and `task` records.
//!
//! Steady state only *appends*; a full atomic rewrite (temp file + rename,
//! mode 0600) happens on first save or when history changed shape.
//! Unknown record types are skipped, so files from newer versions load.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::compaction::{is_summary_message, make_summary_message};
use crate::models::{Message, Role};

pub fn now_iso() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S+00:00")
        .to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UsageEntry {
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

/// The first line of a session file.
///
/// 🦀 `#[serde(flatten)] extra` collects every field this struct does not
/// name (e.g. Python-only `goal`, `daily_mode`) and writes it back
/// unchanged, so round-tripping a Python session never loses data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    #[serde(default)]
    pub workdir: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub updated_at: String,
    #[serde(default)]
    pub usage: HashMap<String, UsageEntry>,
    #[serde(default)]
    pub plan_mode: bool,
    /// Daily mode (non-coding posture); restored on resume.
    #[serde(default)]
    pub daily_mode: bool,
    /// Code Mode (catalog collapsed to `run_code`); restored on resume.
    #[serde(default)]
    pub code_mode: bool,
    /// Active/finished closed-loop goal; restored on resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal: Option<crate::goal::GoalState>,
    #[serde(default)]
    pub tool_intents: bool,
    #[serde(default)]
    pub parent_session: Option<String>,
    #[serde(default)]
    pub fork_at: Option<String>,
    #[serde(default = "main_role")]
    pub role: String,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
    /// Where the file lives (not written to disk).
    #[serde(skip)]
    pub path: Option<PathBuf>,
}

fn main_role() -> String {
    "main".into()
}

impl SessionMeta {
    pub fn new(id: String, workdir: &Path, model: &str) -> Self {
        let now = now_iso();
        Self {
            id,
            workdir: workdir.display().to_string(),
            model: model.into(),
            created_at: now.clone(),
            updated_at: now,
            tool_intents: true,
            role: main_role(),
            ..Default::default()
        }
    }

    fn to_record(&self, kind: &str, drop_id: bool) -> Value {
        let mut v = serde_json::to_value(self).expect("meta serializes");
        let obj = v.as_object_mut().unwrap();
        if drop_id {
            obj.remove("id");
        }
        let mut out = Map::new();
        out.insert("type".into(), json!(kind));
        out.extend(std::mem::take(obj));
        Value::Object(out)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CompactionRecord {
    pub id: String,
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub first_kept_entry_id: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub tokens_before: Option<u64>,
    #[serde(default)]
    pub tokens_after_estimate: Option<u64>,
    #[serde(default = "auto")]
    pub trigger: String,
}

fn auto() -> String {
    "auto".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolIntentRecord {
    pub call_id: String,
    pub name: String,
    #[serde(default = "unsafe_replay")]
    pub replay: String,
    #[serde(default)]
    pub created_at: String,
}

fn unsafe_replay() -> String {
    "unsafe".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueuedMessageRecord {
    pub id: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attachments: Vec<crate::models::Attachment>,
    /// "steer" (injected at the next consistency point) or "follow_up"
    /// (waits until the whole task is done).
    #[serde(default = "steer_mode")]
    pub mode: String,
    /// "user", or "subagent" for a notice delivered to the main agent.
    #[serde(default = "user_source")]
    pub source: String,
    /// Notice ids a subagent delivery carries (dedupe across a crash).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<String>,
}

fn steer_mode() -> String {
    "steer".into()
}
fn user_source() -> String {
    "user".into()
}

/// Everything reconstructed from one file.
#[derive(Debug, Default)]
pub struct LoadedSession {
    pub meta: SessionMeta,
    /// History with the latest compaction applied.
    pub messages: Vec<Message>,
    pub compactions: Vec<CompactionRecord>,
    pub intents: Vec<ToolIntentRecord>,
    /// Messages still queued when the file was last written.
    pub queue: Vec<QueuedMessageRecord>,
    /// Every message line in the file (before compaction selection).
    pub raw_message_ids: Vec<String>,
}

fn record(kind: &str, value: impl Serialize) -> Value {
    let mut v = serde_json::to_value(value).expect("record serializes");
    let mut out = Map::new();
    out.insert("type".into(), json!(kind));
    if let Some(obj) = v.as_object_mut() {
        out.extend(std::mem::take(obj));
    }
    Value::Object(out)
}

pub fn tool_intent_record(call_id: &str, name: &str, replay_safe: bool) -> Value {
    record(
        "tool_intent",
        ToolIntentRecord {
            call_id: call_id.into(),
            name: name.into(),
            replay: if replay_safe { "safe" } else { "unsafe" }.into(),
            created_at: now_iso(),
        },
    )
}

pub fn compaction_record(c: &CompactionRecord) -> Value {
    record("compaction", c)
}

pub fn meta_patch_record(meta: &SessionMeta) -> Value {
    let mut v = meta.to_record("meta_patch", true);
    v.as_object_mut().unwrap().remove("updated_at");
    v
}

pub fn queued_record(item: &QueuedMessageRecord) -> Value {
    let mut v = record("queue", item);
    v["op"] = json!("queued");
    v
}

pub fn dequeued_record(ids: &[String]) -> Value {
    json!({"type": "queue", "op": "dequeued", "ids": ids})
}

#[cfg(unix)]
fn private_open(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
}
#[cfg(not(unix))]
fn private_open(path: &Path) -> std::io::Result<File> {
    File::create(path)
}

/// Atomically (re)write the whole session.
pub fn save_session(
    path: &Path,
    meta: &mut SessionMeta,
    messages: &[Message],
    compactions: &[CompactionRecord],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    meta.updated_at = now_iso();
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::io::BufWriter::new(private_open(&tmp)?);
        writeln!(f, "{}", meta.to_record("meta", false))?;
        for c in compactions {
            writeln!(f, "{}", compaction_record(c))?;
        }
        for m in messages {
            writeln!(f, "{}", serde_json::to_string(m)?)?;
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Write a whole file atomically (temp file + rename) with mode 0600.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}.tmp", crate::models::new_id()));
    {
        let mut f = private_open(&tmp)?;
        f.write_all(bytes)?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Append lines to an existing session (the cheap steady-state write).
pub fn append_lines(path: &Path, lines: &[Value]) -> std::io::Result<()> {
    if !path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Cannot append to missing session file: {}", path.display()),
        ));
    }
    let mut f = OpenOptions::new().append(true).open(path)?;
    let mut buf = String::new();
    for l in lines {
        buf.push_str(&l.to_string());
        buf.push('\n');
    }
    f.write_all(buf.as_bytes())
}

pub fn message_line(m: &Message) -> Value {
    serde_json::to_value(m).expect("message serializes")
}

/// Rewrite just the meta line (e.g. after a title change), keeping every
/// other record; old `meta_patch` lines are dropped (the new meta has them).
pub fn rewrite_meta_line(path: &Path, meta: &mut SessionMeta) -> std::io::Result<()> {
    meta.updated_at = now_iso();
    let tmp = path.with_extension("tmp");
    let src = BufReader::new(File::open(path)?);
    {
        let mut dst = std::io::BufWriter::new(private_open(&tmp)?);
        writeln!(dst, "{}", meta.to_record("meta", false))?;
        for line in src.lines() {
            let line = line?;
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(t)
                && matches!(
                    v.get("type").and_then(Value::as_str),
                    Some("meta" | "meta_patch")
                )
            {
                continue;
            }
            writeln!(dst, "{t}")?;
        }
        dst.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Reconstruct the active range from the latest compaction record.
fn apply_compactions(messages: Vec<Message>, compactions: &[CompactionRecord]) -> Vec<Message> {
    let Some(latest) = compactions.last() else {
        return messages;
    };
    let Some(boundary) = messages
        .iter()
        .position(|m| m.id == latest.first_kept_entry_id)
    else {
        eprintln!(
            "warning: compaction boundary {:?} not found; loading full history",
            latest.first_kept_entry_id
        );
        return messages;
    };
    let head_len = usize::from(messages.first().is_some_and(|m| m.role == Role::System));
    let rebuilt = make_summary_message(&latest.summary);
    let stored = messages[head_len..boundary]
        .iter()
        .rev()
        .find(|m| is_summary_message(m))
        .cloned();
    let summary = match stored {
        Some(s) if latest.summary.is_empty() || s.content == rebuilt.content => s,
        _ => rebuilt,
    };
    let mut out: Vec<Message> = messages[..head_len].to_vec();
    out.push(summary);
    out.extend_from_slice(&messages[boundary..]);
    out
}

fn merge_json(base: &mut Value, patch: &Value) {
    if let (Some(b), Some(p)) = (base.as_object_mut(), patch.as_object()) {
        for (k, v) in p {
            b.insert(k.clone(), v.clone());
        }
    }
}

pub fn load_session_full(path: &Path) -> std::io::Result<LoadedSession> {
    let reader = BufReader::new(File::open(path)?);
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut meta_value: Option<Value> = None;
    let mut patches: Vec<Value> = Vec::new();
    let mut messages = Vec::new();
    let mut compactions = Vec::new();
    let mut intents = Vec::new();
    let mut queued: Vec<QueuedMessageRecord> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let Ok(mut v) = serde_json::from_str::<Value>(t) else {
            eprintln!(
                "warning: skipping malformed line in session {}",
                path.display()
            );
            continue;
        };
        let Some(obj) = v.as_object_mut() else {
            continue;
        };
        let kind = obj
            .remove("type")
            .and_then(|k| k.as_str().map(str::to_string));
        match kind.as_deref() {
            None => {
                if let Ok(m) = serde_json::from_value::<Message>(v) {
                    messages.push(m);
                }
            }
            Some("meta") => meta_value = Some(v),
            Some("meta_patch") => patches.push(v),
            Some("compaction") => {
                if let Ok(c) = serde_json::from_value(v) {
                    compactions.push(c);
                }
            }
            Some("tool_intent") => {
                if let Ok(i) = serde_json::from_value(v) {
                    intents.push(i);
                }
            }
            Some("queue") => match obj.get("op").and_then(Value::as_str) {
                Some("queued") => {
                    if let Ok(q) = serde_json::from_value::<QueuedMessageRecord>(v) {
                        queued.retain(|x| x.id != q.id);
                        queued.push(q);
                    }
                }
                Some("dequeued") => {
                    let ids: Vec<String> = obj
                        .get("ids")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    queued.retain(|x| !ids.contains(&x.id));
                }
                _ => {}
            },
            _ => {} // a record from a newer version, or a subagent board event
        }
    }
    let mut mv = meta_value.unwrap_or_else(|| json!({"id": stem}));
    if mv.get("id").is_none() {
        mv["id"] = json!(stem);
    }
    let id = mv["id"].clone();
    for p in &patches {
        merge_json(&mut mv, p);
    }
    mv["id"] = id;
    let mut meta: SessionMeta = serde_json::from_value(mv).unwrap_or_else(|_| SessionMeta {
        id: stem,
        ..Default::default()
    });
    meta.path = Some(path.to_path_buf());
    let raw_message_ids = messages.iter().map(|m| m.id.clone()).collect();
    Ok(LoadedSession {
        meta,
        messages: apply_compactions(messages, &compactions),
        compactions,
        intents,
        queue: queued,
        raw_message_ids,
    })
}

fn read_meta_line(path: &Path) -> Option<SessionMeta> {
    let mut first = String::new();
    BufReader::new(File::open(path).ok()?)
        .read_line(&mut first)
        .ok()?;
    let v: Value = serde_json::from_str(first.trim()).ok()?;
    if v.get("type").and_then(Value::as_str) != Some("meta") {
        return None;
    }
    serde_json::from_value(v).ok()
}

/// Sessions in `dir`, most recently written first (file mtime wins over the
/// stale meta timestamp because appends never touch the meta line).
pub fn list_sessions(dir: &Path, workdir: Option<&Path>) -> Vec<SessionMeta> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return vec![];
    };
    let wanted = workdir.map(|w| {
        std::fs::canonicalize(w)
            .unwrap_or_else(|_| w.to_path_buf())
            .display()
            .to_string()
    });
    let mut out: Vec<(std::time::SystemTime, SessionMeta)> = Vec::new();
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("jsonl") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let mut meta = read_meta_line(&path).unwrap_or(SessionMeta {
            id: stem,
            ..Default::default()
        });
        if let Some(w) = &wanted
            && &meta.workdir != w
        {
            continue;
        }
        let mtime = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::UNIX_EPOCH);
        meta.updated_at = chrono::DateTime::<chrono::Utc>::from(mtime)
            .format("%Y-%m-%dT%H:%M:%S+00:00")
            .to_string();
        meta.path = Some(path);
        out.push((mtime, meta));
    }
    out.sort_by_key(|e| std::cmp::Reverse(e.0));
    out.into_iter().map(|(_, m)| m).collect()
}

pub fn latest_session(dir: &Path, workdir: Option<&Path>) -> Option<PathBuf> {
    list_sessions(dir, workdir)
        .into_iter()
        .next()
        .and_then(|m| m.path)
}

/// Find by exact id or unique prefix.
pub fn find_session(dir: &Path, prefix: &str) -> anyhow::Result<PathBuf> {
    let mut matches: Vec<String> = std::fs::read_dir(dir)
        .map_err(|_| anyhow::anyhow!("No sessions in {}", dir.display()))?
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            (p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .filter(|s| s.starts_with(prefix))
        .collect();
    matches.sort();
    match matches.len() {
        0 => anyhow::bail!("No session matching '{prefix}'"),
        1 => Ok(dir.join(format!("{}.jsonl", matches[0]))),
        _ if matches.contains(&prefix.to_string()) => Ok(dir.join(format!("{prefix}.jsonl"))),
        _ => anyhow::bail!(
            "Session id prefix '{prefix}' is ambiguous, matches: {}",
            matches.join(", ")
        ),
    }
}

/// First user message, whitespace-collapsed, max 50 chars.
pub fn derive_title(messages: &[Message]) -> String {
    for m in messages {
        if m.role == Role::User && !m.text().is_empty() && !is_summary_message(m) {
            let t = m.text().split_whitespace().collect::<Vec<_>>().join(" ");
            if t.chars().count() > 50 {
                let s: String = t.chars().take(49).collect();
                return format!("{}…", s.trim_end());
            }
            return t;
        }
    }
    String::new()
}

pub fn export_markdown(
    meta: &SessionMeta,
    messages: &[Message],
    path: &Path,
) -> std::io::Result<()> {
    let mut lines = vec![
        format!(
            "# {}",
            if meta.title.is_empty() {
                &meta.id
            } else {
                &meta.title
            }
        ),
        String::new(),
        format!("- workdir: `{}`", meta.workdir),
        format!("- model: `{}`", meta.model),
        format!("- updated: {}", meta.updated_at),
        String::new(),
    ];
    for m in messages {
        match m.role {
            Role::User if !m.text().is_empty() => lines.extend([
                "## User".into(),
                String::new(),
                m.text().into(),
                String::new(),
            ]),
            Role::Assistant if !m.text().is_empty() => lines.extend([
                "## Assistant".into(),
                String::new(),
                m.text().into(),
                String::new(),
            ]),
            _ => {}
        }
    }
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(path, lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_patches_queue_and_unknown_records() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("abc.jsonl");
        let mut meta = SessionMeta::new("abc".into(), dir.path(), "m");
        meta.extra.insert("goal".into(), json!({"text": "keep me"}));
        let msgs = vec![Message::system("s"), Message::user("hello world")];
        save_session(&p, &mut meta, &msgs, &[]).unwrap();
        let mut patched = meta.clone();
        patched.title = "T".into();
        let q = QueuedMessageRecord {
            id: "q1".into(),
            text: "later".into(),
            attachments: vec![],
            mode: "steer".into(),
            source: "user".into(),
            refs: vec![],
        };
        append_lines(
            &p,
            &[
                message_line(&Message::assistant("hi")),
                meta_patch_record(&patched),
                queued_record(&q),
                json!({"type": "future_thing", "x": 1}),
                tool_intent_record("c1", "bash", false),
            ],
        )
        .unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&p)
            .unwrap()
            .write_all(b"{torn")
            .unwrap();
        let loaded = load_session_full(&p).unwrap();
        assert_eq!(loaded.meta.title, "T");
        assert_eq!(loaded.meta.goal.as_ref().unwrap().text, "keep me");
        assert_eq!(loaded.messages.len(), 3);
        assert_eq!(loaded.queue.len(), 1);
        assert_eq!(loaded.intents[0].call_id, "c1");
    }

    #[test]
    fn compaction_selects_active_range() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("s.jsonl");
        let mut meta = SessionMeta::new("s".into(), dir.path(), "m");
        let msgs = vec![
            Message::system("s"),
            Message::user("old"),
            Message::assistant("old reply"),
            Message::user("new"),
        ];
        save_session(&p, &mut meta, &msgs, &[]).unwrap();
        let c = CompactionRecord {
            id: "c".into(),
            first_kept_entry_id: msgs[3].id.clone(),
            summary: "SUM".into(),
            ..Default::default()
        };
        append_lines(&p, &[compaction_record(&c)]).unwrap();
        let loaded = load_session_full(&p).unwrap();
        assert_eq!(loaded.messages.len(), 3);
        assert!(is_summary_message(&loaded.messages[1]));
        assert!(loaded.messages[1].text().contains("SUM"));
        assert_eq!(loaded.messages[2].text(), "new");
    }

    #[test]
    fn loads_a_python_written_session() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("py.jsonl");
        std::fs::write(
            &p,
            r#"{"type": "meta", "id": "py", "workdir": "/w", "model": "glm-4.7", "title": "x", "created_at": "", "updated_at": "", "goal": null, "code_mode": false, "usage": {"glm/glm-4.7": {"requests": 2, "prompt_tokens": 10, "cached_tokens": 0, "total_tokens": 20}}, "plan_mode": false, "daily_mode": false, "tool_intents": true, "parent_session": null, "fork_at": null, "role": "main", "task_id": null}
{"id":"a1","role":"user","content":"hi","tool_calls":null,"tool_call_id":null,"attachments":null,"reasoning":null,"reasoning_signature":null}
{"id":"a2","role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"ls","arguments":{"path":"."}}}],"tool_call_id":null,"attachments":null,"reasoning":"think","reasoning_signature":null}
{"type": "tool_intent", "call_id": "call_1", "name": "ls", "replay": "safe", "created_at": ""}
"#,
        )
        .unwrap();
        let loaded = load_session_full(&p).unwrap();
        assert_eq!(loaded.meta.usage["glm/glm-4.7"].requests, 2);
        assert_eq!(
            loaded.messages[1].tool_calls.as_ref().unwrap()[0]
                .function
                .arguments["path"],
            "."
        );
        assert!(!loaded.meta.daily_mode);
    }

    #[test]
    fn find_by_prefix() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("abc1.jsonl"), "").unwrap();
        std::fs::write(dir.path().join("abd2.jsonl"), "").unwrap();
        assert!(find_session(dir.path(), "abc").is_ok());
        assert!(
            find_session(dir.path(), "ab")
                .unwrap_err()
                .to_string()
                .contains("ambiguous")
        );
    }
}
