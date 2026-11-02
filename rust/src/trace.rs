//! Append-only JSONL trace log (`<session dir>/traces/<id>.trace.jsonl`).
//!
//! Records everything the message history cannot reconstruct afterwards:
//! full LLM request bodies, token usage, tool timing, errors and
//! interruptions. Same format as the Python version, so the
//! `/tracing-viewer` works on traces from either.
//!
//! `log()` only sends the record to a background writer thread, so a
//! multi-MB request body never stalls the UI.
//!
//! 🦀 Rust notes: the writer owns the file; the logger holds the sending
//! half of a `std::sync::mpsc` channel. Dropping the last sender ends the
//! thread's `for` loop — no explicit shutdown flag needed.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

pub const TRACE_SUFFIX: &str = ".trace.jsonl";
pub const TRACE_SCHEMA: u32 = 2;

pub fn trace_path_for(session_file: &Path) -> PathBuf {
    let stem = session_file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    session_file
        .parent()
        .unwrap_or(Path::new("."))
        .join("traces")
        .join(format!("{stem}{TRACE_SUFFIX}"))
}

/// Wall-clock timestamp in the format every record uses (UTC, ms).
pub fn utc_now_iso() -> String {
    chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3f+00:00")
        .to_string()
}

enum Msg {
    Record(Value),
    Flush(mpsc::Sender<()>),
}

/// Cheap to clone; every clone writes to the same file.
#[derive(Clone)]
pub struct TraceLogger {
    pub path: PathBuf,
    tx: Option<Arc<Mutex<mpsc::Sender<Msg>>>>,
}

impl TraceLogger {
    /// A logger that drops everything (tests, failed opens).
    pub fn disabled() -> Self {
        Self {
            path: PathBuf::new(),
            tx: None,
        }
    }

    pub fn open(path: PathBuf) -> Self {
        let file = (|| {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let existed = path.exists();
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?;
            if !existed {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
            }
            Ok::<_, std::io::Error>(f)
        })();
        let Ok(file) = file else {
            return Self { path, tx: None };
        };
        let (tx, rx) = mpsc::channel::<Msg>();
        std::thread::Builder::new()
            .name("limbo-trace-writer".into())
            .spawn(move || {
                let mut w = std::io::BufWriter::new(file);
                for msg in rx {
                    match msg {
                        Msg::Record(v) => {
                            let _ = writeln!(w, "{v}");
                            let _ = w.flush();
                        }
                        Msg::Flush(done) => {
                            let _ = w.flush();
                            let _ = done.send(());
                        }
                    }
                }
            })
            .ok();
        Self {
            path,
            tx: Some(Arc::new(Mutex::new(tx))),
        }
    }

    pub fn enabled(&self) -> bool {
        self.tx.is_some()
    }

    /// Queue one record. Never blocks on disk, never fails.
    pub fn log(&self, kind: &str, fields: Value) {
        let Some(tx) = &self.tx else { return };
        let mut rec = Map::new();
        rec.insert("ts".into(), json!(utc_now_iso()));
        rec.insert("type".into(), json!(kind));
        if let Value::Object(m) = fields {
            for (k, v) in m {
                rec.insert(k, v);
            }
        }
        let _ = tx.lock().unwrap().send(Msg::Record(Value::Object(rec)));
    }

    /// Block until every record logged so far is on disk.
    pub fn flush(&self) {
        let Some(tx) = &self.tx else { return };
        let (done_tx, done_rx) = mpsc::channel();
        if tx.lock().unwrap().send(Msg::Flush(done_tx)).is_ok() {
            let _ = done_rx.recv_timeout(std::time::Duration::from_secs(10));
        }
    }
}

/// Read all records, skipping malformed lines.
pub fn read_trace(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
        .filter(Value::is_object)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_records_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = trace_path_for(&dir.path().join("abc.jsonl"));
        assert!(path.ends_with("traces/abc.trace.jsonl"));
        let t = TraceLogger::open(path.clone());
        t.log("a", json!({"n": 1}));
        t.log("b", json!({"n": 2}));
        t.flush();
        let recs = read_trace(&path);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[1]["type"], "b");
        assert!(recs[0]["ts"].as_str().unwrap().ends_with("+00:00"));
    }
}
