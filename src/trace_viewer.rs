//! `/tracing-viewer`: a loopback HTTP server that shows a session's trace
//! in the browser (port of `limbo/tracing`).
//!
//! The page is the same single self-contained `index.html` the Python
//! version serves; this module provides its JSON API:
//!
//! - `GET /api/state` — the frontend's current session id
//! - `GET /api/sessions` — sessions of the same workdir, newest first
//! - `GET /api/sessions/<id>/tree?cursor=N` — the session tree; when the
//!   trace still has `N` records, `{"unchanged": true}` (live polling)
//! - `GET /api/sessions/<id>/records/<offset>[?diff=1]` — one full record
//!
//! Traces hold full prompts and tool output, so access is locked down:
//! bound to 127.0.0.1, every request needs a random per-process token
//! (passed once in the URL, then kept in an HttpOnly cookie), and the Host
//! header must name the loopback address (blocks DNS rebinding).
//!
//! 🦀 This uses plain `std::net` and one OS thread per connection instead
//! of tokio: the server is tiny, rarely busy, and must never compete with
//! the TUI's event loop.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

use crate::llm::usage::normalize_usage;
use crate::session::{find_session, list_sessions};
use crate::trace::trace_path_for;

const PAGE: &str = include_str!("../assets/trace_viewer.html");
const COOKIE: &str = "limbo_trace";
const CSP: &str = "default-src 'none'; script-src 'unsafe-inline'; \
                   style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:";
pub const USAGE: &str = "/tracing-viewer [会话 id 前缀|off]";

// ---------------------------------------------------------------------------
// index: byte offsets + summaries

const BULKY: [&str; 6] = [
    "body",
    "output",
    "content",
    "reasoning",
    "traceback",
    "arguments",
];
const PREVIEW_CHARS: usize = 200;

fn preview(v: &Value) -> String {
    let text = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if text.chars().count() <= PREVIEW_CHARS {
        text
    } else {
        let mut s: String = text.chars().take(PREVIEW_CHARS).collect();
        s.push('…');
        s
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.0),
    }
}

/// The record without bulky fields, with short previews in their place.
pub fn summarize(record: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    for (key, value) in record {
        if key == "body" {
            if let Value::Object(body) = value {
                if let Some(m) = body.get("model").filter(|m| !m.is_null()) {
                    out.insert("model".into(), m.clone());
                }
                if let Some(Value::Array(msgs)) = body.get("messages").or_else(|| body.get("input"))
                {
                    out.insert("message_count".into(), json!(msgs.len()));
                }
            }
            continue;
        }
        if BULKY.contains(&key.as_str()) {
            if truthy(value) {
                out.insert(format!("{key}_preview"), json!(preview(value)));
            }
            continue;
        }
        if key == "tool_calls"
            && let Value::Array(calls) = value
        {
            let slim: Vec<Value> = calls
                .iter()
                .filter_map(|c| c.as_object())
                .map(|c| {
                    Value::Object(
                        c.iter()
                            .filter(|(k, _)| *k != "arguments")
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    )
                })
                .collect();
            out.insert(key.clone(), Value::Array(slim));
            continue;
        }
        out.insert(key.clone(), value.clone());
    }
    out
}

#[derive(Clone)]
pub struct Indexed {
    pub offset: u64,
    pub summary: Map<String, Value>,
}

/// Incremental index of one trace file: `refresh` only reads new bytes.
pub struct TraceIndex {
    path: PathBuf,
    records: Vec<Indexed>,
    offsets: HashSet<u64>,
    scanned: u64,
}

impl TraceIndex {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            records: vec![],
            offsets: HashSet::new(),
            scanned: 0,
        }
    }

    pub fn refresh(&mut self) -> usize {
        let Ok(size) = std::fs::metadata(&self.path).map(|m| m.len()) else {
            return 0;
        };
        if size < self.scanned {
            // Rewritten or truncated underneath us: start over.
            self.records.clear();
            self.offsets.clear();
            self.scanned = 0;
        }
        if size == self.scanned {
            return 0;
        }
        let mut chunk = vec![];
        let read = std::fs::File::open(&self.path).and_then(|mut f| {
            f.seek(SeekFrom::Start(self.scanned))?;
            f.take(size - self.scanned).read_to_end(&mut chunk)
        });
        if read.is_err() {
            return 0;
        }
        let mut added = 0;
        let mut pos = 0usize;
        while let Some(nl) = chunk[pos..].iter().position(|b| *b == b'\n') {
            let line = &chunk[pos..pos + nl];
            let offset = self.scanned + pos as u64;
            pos += nl + 1;
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            if let Ok(Value::Object(rec)) = serde_json::from_slice::<Value>(line) {
                self.records.push(Indexed {
                    offset,
                    summary: summarize(&rec),
                });
                self.offsets.insert(offset);
                added += 1;
            }
        }
        // A trailing line without newline is still being written.
        self.scanned += pos as u64;
        added
    }

    pub fn records(&self) -> &[Indexed] {
        &self.records
    }

    /// The full record at `offset`; only indexed offsets are served.
    pub fn read(&self, offset: u64) -> Option<Value> {
        if !self.offsets.contains(&offset) {
            return None;
        }
        let mut f = std::fs::File::open(&self.path).ok()?;
        f.seek(SeekFrom::Start(offset)).ok()?;
        let mut line = String::new();
        BufReader::new(f).read_line(&mut line).ok()?;
        serde_json::from_str::<Value>(&line)
            .ok()
            .filter(Value::is_object)
    }

    /// Offset of the agent `llm_request` before the one at `offset`
    /// (compaction requests skipped).
    pub fn previous_request(&self, offset: u64) -> Option<u64> {
        let mut prev = None;
        for item in &self.records {
            if item.offset >= offset {
                break;
            }
            let s = &item.summary;
            if s.get("type").and_then(Value::as_str) == Some("llm_request")
                && s.get("kind").and_then(Value::as_str) != Some("compaction")
            {
                prev = Some(item.offset);
            }
        }
        prev
    }
}

// ---------------------------------------------------------------------------
// model: flat records → session tree (Session / Trace / Generation / Span)

const ERROR_TYPES: [&str; 4] = [
    "llm_error",
    "error",
    "compaction_error",
    "session_save_error",
];

fn get(s: &Map<String, Value>, k: &str) -> Value {
    s.get(k).cloned().unwrap_or(Value::Null).to_owned()
}

fn start_of(s: &Map<String, Value>) -> Value {
    if let Some(Value::String(started)) = s.get("started_at") {
        return json!(started);
    }
    let end = s
        .get("ts")
        .and_then(Value::as_str)
        .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok());
    let dur = s.get("duration").and_then(Value::as_f64);
    match (end, dur) {
        (Some(end), Some(d)) => {
            let start = end - chrono::Duration::milliseconds((d * 1000.0).round() as i64);
            json!(start.format("%Y-%m-%dT%H:%M:%S%.3f%:z").to_string())
        }
        _ => get(s, "ts"),
    }
}

fn tokens(usage: &Value) -> Map<String, Value> {
    let t = normalize_usage(Some(usage));
    let mut completion = ["completion_tokens", "output_tokens"]
        .iter()
        .find_map(|k| usage.get(*k).and_then(Value::as_u64));
    if completion.is_none()
        && let (Some(total), Some(prompt)) = (t.total_tokens, t.prompt_tokens)
        && prompt > 0
    {
        completion = Some(total.saturating_sub(prompt));
    }
    let mut m = Map::new();
    m.insert("input".into(), json!(t.prompt_tokens));
    m.insert("output".into(), json!(completion));
    m.insert("cached".into(), json!(t.cached_tokens));
    m.insert("total".into(), json!(t.total_tokens));
    m
}

fn empty_totals() -> Value {
    json!({"tokens": {}, "generations": 0, "tools": 0, "errors": 0,
           "llm_seconds": 0.0, "tool_seconds": 0.0})
}

fn add_tokens(totals: &mut Value, toks: &Map<String, Value>) {
    for (k, v) in toks {
        if let Some(n) = v.as_u64().filter(|n| *n > 0) {
            let cur = totals["tokens"][k].as_u64().unwrap_or(0);
            totals["tokens"][k] = json!(cur + n);
        }
    }
}

fn bump(totals: &mut Value, key: &str) {
    totals[key] = json!(totals[key].as_u64().unwrap_or(0) + 1);
}

fn add_secs(totals: &mut Value, key: &str, d: &Value) {
    if let Some(d) = d.as_f64() {
        totals[key] = json!(totals[key].as_f64().unwrap_or(0.0) + d);
    }
}

fn without(s: &Map<String, Value>, drop: &[&str]) -> Value {
    Value::Object(
        s.iter()
            .filter(|(k, _)| !drop.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )
}

fn update(span: &mut Value, fields: Value) {
    if let (Value::Object(m), Value::Object(f)) = (span, fields) {
        for (k, v) in f {
            m.insert(k, v);
        }
    }
}

#[derive(Default)]
struct Builder {
    session: Map<String, Value>,
    turns: Vec<Value>,
    events: Vec<Value>,
    totals: Value,
    open: Option<usize>,
    /// (iteration, kind) → span index in the open turn.
    generations: HashMap<String, usize>,
    /// tool call id → span index in the open turn.
    tools: HashMap<String, usize>,
}

impl Builder {
    fn new() -> Self {
        Self {
            totals: empty_totals(),
            ..Default::default()
        }
    }

    fn open_turn(&mut self, ts: Value, input: Value, offset: Value) -> usize {
        self.turns.push(json!({
            "index": self.turns.len() + 1, "input": input, "input_offset": offset,
            "start": ts, "end": null, "duration": null, "status": "running",
            "spans": [], "totals": empty_totals(),
        }));
        let i = self.turns.len() - 1;
        self.open = Some(i);
        self.generations.clear();
        self.tools.clear();
        i
    }

    fn turn(&mut self, s: &Map<String, Value>) -> usize {
        match self.open {
            Some(i) => i,
            None => self.open_turn(get(s, "ts"), Value::Null, Value::Null),
        }
    }

    fn spans(&mut self, t: usize) -> &mut Vec<Value> {
        self.turns[t]["spans"].as_array_mut().unwrap()
    }

    fn push_span(&mut self, t: usize, span: Value) -> usize {
        let spans = self.spans(t);
        spans.push(span);
        spans.len() - 1
    }

    fn event(&mut self, s: &Map<String, Value>, offset: u64) {
        let ev = json!({
            "kind": "event", "type": get(s, "type"), "ts": get(s, "ts"),
            "offset": offset, "summary": without(s, &["ts", "type"]),
        });
        match self.open {
            Some(t) => {
                self.push_span(t, ev);
            }
            None => self.events.push(ev),
        }
    }

    fn error(&mut self, turn: Option<usize>) {
        bump(&mut self.totals, "errors");
        if let Some(t) = turn {
            bump(&mut self.turns[t]["totals"], "errors");
        }
    }

    fn fail_pending(&mut self, s: &Map<String, Value>, status: &str) {
        let Some(t) = self.open else { return };
        let ts = get(s, "ts");
        for span in self.spans(t) {
            if span["status"] == "running" {
                span["status"] = json!(status);
                span["end"] = ts.clone();
            }
        }
    }

    fn add(&mut self, item: &Indexed) {
        let (s, offset) = (&item.summary, item.offset);
        let kind = s.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "session_start" => {
                if let Some(t) = self.open {
                    self.turns[t]["status"] = json!("unfinished");
                    self.fail_pending(s, "unfinished");
                    self.open = None;
                }
                if self.session.is_empty() {
                    if let Value::Object(m) = without(s, &["type"]) {
                        self.session = m;
                    }
                    self.session.insert("offset".into(), json!(offset));
                }
                self.event(s, offset);
            }
            "user_message"
                if !s.get("steer").is_some_and(truthy)
                    && !s.get("followup").is_some_and(truthy) =>
            {
                if let Some(t) = self.open {
                    self.turns[t]["status"] = json!("unfinished");
                }
                self.open_turn(get(s, "ts"), get(s, "content_preview"), json!(offset));
            }
            "llm_request" => self.request(s, offset),
            "llm_response" => self.response(s, offset),
            "tool_call" => self.tool_call(s, offset),
            "tool_result" => self.tool_result(s, offset),
            "turn_end" => self.turn_end(s),
            _ => {
                if ERROR_TYPES.contains(&kind) {
                    self.error(self.open);
                    self.fail_pending(s, "error");
                } else if kind == "turn_interrupted" {
                    self.fail_pending(s, "interrupted");
                } else if kind == "compaction"
                    && let Some(t) = self.open
                {
                    self.push_span(
                        t,
                        json!({
                            "kind": "compaction", "id": format!("c{offset}"), "name": "compaction",
                            "start": start_of(s), "end": get(s, "ts"), "duration": get(s, "duration"),
                            "status": "ok", "offset": offset, "summary": without(s, &["ts", "type"]),
                        }),
                    );
                    return;
                }
                self.event(s, offset);
            }
        }
    }

    fn gen_key(s: &Map<String, Value>) -> String {
        format!("{}|{}", get(s, "iteration"), get(s, "kind"))
    }

    fn gen_name(s: &Map<String, Value>) -> &'static str {
        if s.get("kind").and_then(Value::as_str) == Some("compaction") {
            "compaction"
        } else {
            "llm"
        }
    }

    fn request(&mut self, s: &Map<String, Value>, offset: u64) {
        let t = self.turn(s);
        let key = Self::gen_key(s);
        let existing = self
            .generations
            .get(&key)
            .copied()
            .filter(|i| self.turns[t]["spans"][*i]["status"] == "running");
        let i = match existing {
            Some(i) => i,
            None => {
                let span = json!({
                    "kind": "generation", "id": format!("g{offset}"), "name": Self::gen_name(s),
                    "iteration": get(s, "iteration"), "model": get(s, "model"),
                    "message_count": get(s, "message_count"), "start": get(s, "ts"),
                    "end": null, "duration": null, "status": "running", "attempts": [],
                    "response_offset": null, "tokens": {},
                });
                let i = self.push_span(t, span);
                self.generations.insert(key, i);
                i
            }
        };
        self.spans(t)[i]["attempts"]
            .as_array_mut()
            .unwrap()
            .push(json!(offset));
    }

    fn response(&mut self, s: &Map<String, Value>, offset: u64) {
        let t = self.turn(s);
        let key = Self::gen_key(s);
        let i = match self.generations.get(&key).copied() {
            Some(i) => i,
            None => {
                let span = json!({
                    "kind": "generation", "id": format!("g{offset}"), "name": Self::gen_name(s),
                    "iteration": get(s, "iteration"), "attempts": [],
                });
                let i = self.push_span(t, span);
                self.generations.insert(key, i);
                i
            }
        };
        let toks = tokens(&get(s, "usage"));
        let duration = get(s, "duration");
        let span = &mut self.spans(t)[i];
        let start = match start_of(s) {
            Value::Null => span.get("start").cloned().unwrap_or(Value::Null),
            v => v,
        };
        let tool_calls = match get(s, "tool_calls") {
            Value::Null => json!([]),
            v => v,
        };
        update(
            span,
            json!({
                "start": start, "end": get(s, "ts"), "duration": duration,
                "ttft": get(s, "ttft"), "finish_reason": get(s, "finish_reason"),
                "tokens": toks, "tool_calls": tool_calls,
                "output_preview": get(s, "content_preview"), "status": "ok",
                "response_offset": offset,
            }),
        );
        for totals in [&mut self.totals, &mut self.turns[t]["totals"]] {
            bump(totals, "generations");
            add_tokens(totals, &toks);
            add_secs(totals, "llm_seconds", &duration);
        }
    }

    fn tool_call(&mut self, s: &Map<String, Value>, offset: u64) {
        let t = self.turn(s);
        let span = json!({
            "kind": "tool", "id": format!("t{offset}"), "call_id": get(s, "id"),
            "name": get(s, "name"), "iteration": get(s, "iteration"), "start": get(s, "ts"),
            "end": null, "duration": null, "status": "running", "call_offset": offset,
            "result_offset": null, "arguments_preview": get(s, "arguments_preview"),
            "parent_call_id": get(s, "parent_call_id"),
        });
        let i = self.push_span(t, span);
        self.tools.insert(get(s, "id").to_string(), i);
    }

    fn tool_result(&mut self, s: &Map<String, Value>, offset: u64) {
        let t = self.turn(s);
        let i = match self.tools.remove(&get(s, "id").to_string()) {
            Some(i) => i,
            None => self.push_span(
                t,
                json!({"kind": "tool", "id": format!("t{offset}"), "name": get(s, "name"),
                       "call_offset": null}),
            ),
        };
        let duration = get(s, "duration");
        let failed = s.get("success") == Some(&Value::Bool(false));
        update(
            &mut self.spans(t)[i],
            json!({
                "start": start_of(s), "end": get(s, "ts"), "duration": duration,
                "status": if failed { "error" } else { "ok" }, "result_offset": offset,
                "output_preview": get(s, "output_preview"), "error": get(s, "error"),
            }),
        );
        // Sub-call time is already inside its run_code parent's duration.
        if s.get("parent_call_id").is_none_or(Value::is_null) {
            for totals in [&mut self.totals, &mut self.turns[t]["totals"]] {
                bump(totals, "tools");
                add_secs(totals, "tool_seconds", &duration);
            }
        }
        if failed {
            self.error(Some(t));
        }
    }

    fn turn_end(&mut self, s: &Map<String, Value>) {
        let t = self.turn(s);
        self.turns[t]["end"] = get(s, "ts");
        self.turns[t]["duration"] = get(s, "duration");
        if s.get("started_at").is_some_and(truthy) {
            self.turns[t]["start"] = get(s, "started_at");
        }
        let status = s
            .get("status")
            .and_then(Value::as_str)
            .filter(|x| !x.is_empty())
            .unwrap_or("completed")
            .to_string();
        self.turns[t]["status"] = json!(status);
        self.fail_pending(
            s,
            if status == "interrupted" {
                "interrupted"
            } else {
                "unfinished"
            },
        );
        self.open = None;
    }
}

pub fn build_tree(records: &[Indexed]) -> Value {
    let mut b = Builder::new();
    for r in records {
        b.add(r);
    }
    json!({
        "session": b.session, "events": b.events, "turns": b.turns, "totals": b.totals,
        "record_count": records.len(), "live": b.open.is_some(),
    })
}

fn request_messages(rec: &Value) -> Vec<Value> {
    let body = &rec["body"];
    body.get("messages")
        .or_else(|| body.get("input"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Messages new in `current`'s request since `previous`'s (the tail past
/// the longest common prefix).
pub fn request_diff(current: &Value, previous: Option<&Value>) -> Value {
    let msgs = request_messages(current);
    let Some(prev) = previous else {
        return json!({"prefix": 0, "total": msgs.len(), "messages": msgs});
    };
    let old = request_messages(prev);
    let prefix = old.iter().zip(&msgs).take_while(|(a, b)| a == b).count();
    json!({"prefix": prefix, "total": msgs.len(), "messages": &msgs[prefix..]})
}

// ---------------------------------------------------------------------------
// server

struct Shared {
    session_dir: PathBuf,
    workdir: Option<PathBuf>,
    current: Mutex<String>,
    token: String,
    port: u16,
    indexes: Mutex<HashMap<String, TraceIndex>>,
    stop: AtomicBool,
}

fn valid_id(id: &str) -> bool {
    let mut chars = id.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
        && !id.contains("..")
}

impl Shared {
    fn trace_file(&self, id: &str) -> Option<PathBuf> {
        valid_id(id).then(|| trace_path_for(&self.session_dir.join(format!("{id}.jsonl"))))
    }

    fn current(&self) -> String {
        self.current.lock().unwrap().clone()
    }

    fn sessions(&self) -> Value {
        let current = self.current();
        let mut out: Vec<Value> = list_sessions(&self.session_dir, self.workdir.as_deref())
            .into_iter()
            .map(|m| {
                let has = self.trace_file(&m.id).is_some_and(|p| p.exists());
                json!({
                    "id": m.id, "title": m.title, "model": m.model,
                    "updated_at": m.updated_at, "created_at": m.created_at,
                    "current": m.id == current, "has_trace": has,
                })
            })
            .collect();
        if !current.is_empty() && !out.iter().any(|s| s["id"] == current.as_str()) {
            // Brand-new session: no session file saved yet.
            out.insert(
                0,
                json!({"id": current, "title": "", "current": true, "has_trace": true}),
            );
        }
        Value::Array(out)
    }

    /// Run `f` on the refreshed index of `id`.
    fn with_index<T>(&self, id: &str, f: impl FnOnce(&TraceIndex) -> T) -> Option<T> {
        let path = self.trace_file(id)?;
        let mut all = self.indexes.lock().unwrap();
        let idx = all
            .entry(id.to_string())
            .or_insert_with(|| TraceIndex::new(path));
        idx.refresh();
        Some(f(idx))
    }
}

struct Response {
    status: u16,
    body: Vec<u8>,
    content_type: &'static str,
    extra: Vec<(&'static str, String)>,
}

fn json_resp(status: u16, v: Value) -> Response {
    Response {
        status,
        body: v.to_string().into_bytes(),
        content_type: "application/json; charset=utf-8",
        extra: vec![],
    }
}

fn error_resp(status: u16, msg: &str) -> Response {
    json_resp(status, json!({"error": msg}))
}

fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('=').or(Some((kv, ""))))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

fn cookie_token(cookie: &str) -> Option<&str> {
    cookie
        .split(';')
        .filter_map(|p| p.trim().split_once('='))
        .find(|(k, _)| *k == COOKIE)
        .map(|(_, v)| v.trim_matches('"'))
}

/// Constant-time comparison (the token is a secret).
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .fold(0u8, |acc, (x, y)| acc | (x ^ y))
            == 0
}

fn route(sh: &Shared, method: &str, target: &str, headers: &HashMap<String, String>) -> Response {
    let host = headers
        .get("host")
        .map(|h| h.to_lowercase())
        .unwrap_or_default();
    if host != format!("127.0.0.1:{}", sh.port) && host != format!("localhost:{}", sh.port) {
        return error_resp(403, "bad host");
    }
    if method != "GET" {
        return error_resp(405, "method not allowed");
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let url_token = query_param(query, "token").filter(|t| !t.is_empty());
    let given = url_token
        .or_else(|| headers.get("cookie").and_then(|c| cookie_token(c)))
        .unwrap_or("");
    if !same(given, &sh.token) {
        return error_resp(401, "missing or wrong token; reopen with /tracing-viewer");
    }
    if path == "/" {
        if let Some(tok) = url_token {
            // Swap the token for an HttpOnly cookie so it leaves the
            // address bar; the #session fragment survives the redirect.
            return Response {
                status: 302,
                body: vec![],
                content_type: "text/plain; charset=utf-8",
                extra: vec![
                    (
                        "Set-Cookie",
                        format!("{COOKIE}={tok}; HttpOnly; SameSite=Strict; Path=/"),
                    ),
                    ("Location", "/".into()),
                ],
            };
        }
        return Response {
            status: 200,
            body: PAGE.as_bytes().to_vec(),
            content_type: "text/html; charset=utf-8",
            extra: vec![],
        };
    }
    if path == "/api/state" {
        return json_resp(200, json!({"current": sh.current()}));
    }
    if path == "/api/sessions" {
        return json_resp(200, json!({"sessions": sh.sessions()}));
    }
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    match parts.as_slice() {
        ["api", "sessions", id, "tree"] => {
            let cursor = query_param(query, "cursor").and_then(|c| c.parse::<usize>().ok());
            let current = *id == sh.current();
            let r = sh.with_index(id, |idx| {
                let n = idx.records().len();
                if cursor == Some(n) {
                    return json!({"unchanged": true, "record_count": n});
                }
                let mut tree = build_tree(idx.records());
                tree["current"] = json!(current);
                tree
            });
            match r {
                Some(v) => json_resp(200, v),
                None => error_resp(404, "unknown session"),
            }
        }
        ["api", "sessions", id, "records", off] => {
            let Ok(offset) = off.parse::<u64>() else {
                return error_resp(404, "not found");
            };
            let diff = query_param(query, "diff").is_some_and(|d| !d.is_empty());
            let r = sh.with_index(id, |idx| {
                let rec = idx.read(offset)?;
                let mut payload = json!({"record": rec});
                if diff && rec["type"] == "llm_request" {
                    let prev = if rec["kind"] == "compaction" {
                        None
                    } else {
                        idx.previous_request(offset).and_then(|o| idx.read(o))
                    };
                    payload["diff"] = request_diff(&rec, prev.as_ref());
                }
                Some(payload)
            });
            match r.flatten() {
                Some(v) => json_resp(200, v),
                None => error_resp(404, "unknown record"),
            }
        }
        _ => error_resp(404, "not found"),
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        302 => "Found",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    }
}

fn handle(sh: &Shared, stream: TcpStream) -> std::io::Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }
    let mut parts = request_line.split_whitespace();
    let resp = match (parts.next(), parts.next()) {
        (Some(m), Some(t)) => route(sh, m, t, &headers),
        _ => error_resp(400, "bad request"),
    };
    let mut out = stream;
    let mut head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\n\
         X-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nX-Frame-Options: DENY\r\n\
         Content-Security-Policy: {CSP}\r\nConnection: close\r\n",
        resp.status,
        reason(resp.status),
        resp.content_type,
        resp.body.len()
    );
    for (k, v) in &resp.extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    out.write_all(head.as_bytes())?;
    out.write_all(&resp.body)?;
    out.flush()
}

fn random_token() -> String {
    use rand::Rng;
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut rng = rand::rng();
    (0..24)
        .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())] as char)
        .collect()
}

struct Server {
    shared: Arc<Shared>,
}

impl Server {
    fn start(
        session_dir: PathBuf,
        workdir: Option<PathBuf>,
        current: String,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let shared = Arc::new(Shared {
            session_dir,
            workdir,
            current: Mutex::new(current),
            token: random_token(),
            port,
            indexes: Mutex::new(HashMap::new()),
            stop: AtomicBool::new(false),
        });
        let sh = shared.clone();
        std::thread::Builder::new()
            .name("limbo-trace-viewer".into())
            .spawn(move || {
                for stream in listener.incoming() {
                    if sh.stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let sh = sh.clone();
                    std::thread::spawn(move || {
                        let _ = handle(&sh, stream);
                    });
                }
            })?;
        Ok(Self { shared })
    }

    fn url(&self, session: &str) -> String {
        let mut url = format!(
            "http://127.0.0.1:{}/?token={}",
            self.shared.port, self.shared.token
        );
        if !session.is_empty() {
            url.push('#');
            url.push_str(session);
        }
        url
    }

    fn stop(&self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        // Wake the blocking accept so the thread sees the flag.
        let _ = TcpStream::connect(("127.0.0.1", self.shared.port));
    }
}

/// Whether launching a browser can work without hijacking the terminal.
pub fn can_open_browser() -> bool {
    if std::env::var_os("SSH_CONNECTION").is_some() || std::env::var_os("SSH_TTY").is_some() {
        return false;
    }
    if cfg!(target_os = "linux") {
        return std::env::var_os("DISPLAY").is_some()
            || std::env::var_os("WAYLAND_DISPLAY").is_some();
    }
    true
}

fn open_browser(url: &str) -> bool {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    std::process::Command::new(cmd)
        .arg(url)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .is_ok()
}

/// The `/tracing-viewer` command, shared by the TUI and headless frontends.
pub struct Viewer {
    session_dir: PathBuf,
    workdir: Option<PathBuf>,
    server: Option<Server>,
}

impl Viewer {
    pub fn new(session_dir: PathBuf, workdir: Option<PathBuf>) -> Self {
        Self {
            session_dir,
            workdir,
            server: None,
        }
    }

    /// Execute the command; returns the line to show the user. `current` is
    /// the frontend's current session id (None keeps the last one known).
    pub fn command(&mut self, arg: &str, current: Option<&str>) -> String {
        let arg = arg.trim();
        if arg == "off" {
            if self.server.is_none() {
                return "Tracing viewer 未在运行".into();
            }
            self.stop();
            return "已关闭 tracing viewer".into();
        }
        if let (Some(s), Some(id)) = (&self.server, current) {
            *s.shared.current.lock().unwrap() = id.to_string();
        }
        let mut target = match (current, &self.server) {
            (Some(id), _) => id.to_string(),
            (None, Some(s)) => s.shared.current(),
            (None, None) => String::new(),
        };
        if !arg.is_empty() {
            match find_session(&self.session_dir, arg) {
                Ok(p) => target = stem(&p),
                Err(e) if e.to_string().contains("ambiguous") => {
                    return format!("会话前缀 {arg} 不唯一：{e}");
                }
                Err(_) => return format!("找不到会话：{arg}（用法：{USAGE}）"),
            }
        }
        if self.server.is_none() {
            match Server::start(
                self.session_dir.clone(),
                self.workdir.clone(),
                current.unwrap_or(&target).to_string(),
            ) {
                Ok(s) => self.server = Some(s),
                Err(e) => return format!("Tracing viewer 启动失败：{e}"),
            }
        }
        let server = self.server.as_ref().unwrap();
        let url = server.url(&target);
        if can_open_browser() && open_browser(&url) {
            return format!("Tracing viewer 已在浏览器打开：{url}");
        }
        let port = server.shared.port;
        format!(
            "Tracing viewer：{url}\n（未能自动打开浏览器，请复制链接；远程机器可先 ssh -L {port}:127.0.0.1:{port} 转发端口）"
        )
    }

    pub fn stop(&mut self) {
        if let Some(s) = self.server.take() {
            s.stop();
        }
    }
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(offset: u64, v: Value) -> Indexed {
        Indexed {
            offset,
            summary: summarize(v.as_object().unwrap()),
        }
    }

    #[test]
    fn summary_drops_bulky_fields() {
        let s = summarize(
            json!({"type": "llm_request", "body": {"model": "m", "messages": [1, 2]},
                   "output": "x".repeat(300)})
            .as_object()
            .unwrap(),
        );
        assert_eq!(s["model"], "m");
        assert_eq!(s["message_count"], 2);
        assert!(s["output_preview"].as_str().unwrap().ends_with('…'));
        assert!(!s.contains_key("body"));
    }

    #[test]
    fn tree_groups_turns_generations_and_tools() {
        let records = vec![
            rec(
                0,
                json!({"type": "session_start", "ts": "2026-01-01T00:00:00.000+00:00"}),
            ),
            rec(
                10,
                json!({"type": "user_message", "content": "hi", "ts": "2026-01-01T00:00:01.000+00:00"}),
            ),
            rec(
                20,
                json!({"type": "llm_request", "iteration": 1, "body": {"messages": []}}),
            ),
            rec(
                30,
                json!({"type": "llm_response", "iteration": 1, "duration": 1.5,
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                           "ts": "2026-01-01T00:00:03.000+00:00"}),
            ),
            rec(40, json!({"type": "tool_call", "id": "c1", "name": "bash"})),
            rec(
                50,
                json!({"type": "tool_result", "id": "c1", "name": "bash", "success": false, "duration": 0.5}),
            ),
            rec(60, json!({"type": "turn_end", "status": "completed"})),
        ];
        let tree = build_tree(&records);
        assert_eq!(tree["record_count"], 7);
        assert_eq!(tree["live"], false);
        let turn = &tree["turns"][0];
        assert_eq!(turn["input"], "hi");
        assert_eq!(turn["status"], "completed");
        let spans = turn["spans"].as_array().unwrap();
        assert_eq!(spans[0]["kind"], "generation");
        assert_eq!(spans[0]["tokens"]["output"], 5);
        assert_eq!(spans[0]["start"], "2026-01-01T00:00:01.500+00:00");
        assert_eq!(spans[1]["status"], "error");
        assert_eq!(tree["totals"]["errors"], 1);
        assert_eq!(tree["totals"]["tokens"]["total"], 15);
    }

    #[test]
    fn diff_returns_new_tail() {
        let a = json!({"body": {"messages": [1, 2]}});
        let b = json!({"body": {"messages": [1, 2, 3]}});
        let d = request_diff(&b, Some(&a));
        assert_eq!(d["prefix"], 2);
        assert_eq!(d["messages"], json!([3]));
    }

    #[test]
    fn server_requires_token_and_serves_tree() {
        let dir = tempfile::tempdir().unwrap();
        let traces = dir.path().join("traces");
        std::fs::create_dir_all(&traces).unwrap();
        std::fs::write(
            traces.join("s1.trace.jsonl"),
            "{\"type\":\"user_message\",\"content\":\"x\"}\n{\"type\":\"turn_end\"}\n",
        )
        .unwrap();
        let server = Server::start(dir.path().into(), None, "s1".into()).unwrap();
        let port = server.shared.port;
        let get = |path: &str, cookie: Option<&str>| {
            let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
            let c = cookie
                .map(|c| format!("Cookie: {c}\r\n"))
                .unwrap_or_default();
            write!(
                s,
                "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{c}\r\n"
            )
            .unwrap();
            let mut out = String::new();
            s.read_to_string(&mut out).unwrap();
            out
        };
        assert!(get("/api/state", None).starts_with("HTTP/1.1 401"));
        let tok = server.shared.token.clone();
        let redirect = get(&format!("/?token={tok}"), None);
        assert!(redirect.starts_with("HTTP/1.1 302"));
        assert!(redirect.contains("HttpOnly"));
        let cookie = format!("{COOKIE}={tok}");
        let tree = get("/api/sessions/s1/tree", Some(&cookie));
        assert!(tree.contains("\"record_count\":2"), "{tree}");
        assert!(get("/api/sessions/s1/tree?cursor=2", Some(&cookie)).contains("unchanged"));
        assert!(get("/api/sessions/../tree", Some(&cookie)).contains("404"));
        server.stop();
    }
}
