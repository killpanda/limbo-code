//! The execution journal: one `action` record per step of every action a
//! session runs, appended to the session file next to its messages.
//!
//! A tool call writes `started` (or `denied`) before anything runs, then
//! `completed` or `failed` as soon as its result is in. A goal verify run
//! does the same with kind `verify`. On resume the pair tells crash
//! recovery what happened to a call whose result message never reached
//! the file:
//!
//! | journal                 | meaning                                      |
//! |-------------------------|----------------------------------------------|
//! | nothing                 | it never started                             |
//! | `started` only          | unknown: it may have partially run           |
//! | `started` + finished    | it ran to the end; only the result was lost  |
//!
//! Files written before this journal carry `tool_intent` records instead;
//! they load as `started`.
//!
//! 🦀 Rust notes
//! - `#[serde(rename_all = "snake_case")]` on an enum writes its variants
//!   as lowercase strings (`"completed"`), which keeps the JSONL readable.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::capability::{Capability, Provider};
use crate::effect::Effect;

/// What kind of action a record is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// A model tool call (`id` = the tool call id).
    Tool,
    /// A `/goal` verify command run (`id` = `verify-<round>`).
    Verify,
}

/// One step in an action's life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionPhase {
    /// Admitted and about to run: any side effect happens after this.
    Started,
    /// Refused by admission (or an unknown tool); it never ran.
    Denied,
    Completed,
    Failed,
}

impl ActionPhase {
    pub fn is_finished(self) -> bool {
        matches!(self, ActionPhase::Completed | ActionPhase::Failed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionRecord {
    pub id: String,
    pub kind: ActionKind,
    pub name: String,
    pub phase: ActionPhase,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effect: Option<Effect>,
    /// Who carried it out (absent for unknown tools and legacy records).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Provider>,
    /// A short note: the denial reason, a verify exit code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(default)]
    pub at: String,
}

impl ActionRecord {
    pub fn new(kind: ActionKind, id: &str, name: &str, phase: ActionPhase) -> Self {
        Self {
            id: id.into(),
            kind,
            name: name.into(),
            phase,
            effect: None,
            provider: None,
            detail: None,
            at: crate::session::now_iso(),
        }
    }
    pub fn with_effect(mut self, effect: Option<Effect>) -> Self {
        self.effect = effect;
        self
    }
    /// Effect and provider of a capability (`None` = an unknown tool).
    pub fn with_capability(mut self, cap: Option<Capability>) -> Self {
        self.effect = cap.map(|c| c.effect);
        self.provider = cap.map(|c| c.provider);
        self
    }
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
    /// The JSONL line (`"type": "action"`).
    pub fn to_line(&self) -> Value {
        let mut v = serde_json::to_value(self).expect("action serializes");
        let mut out = serde_json::Map::new();
        out.insert("type".into(), "action".into());
        if let Some(obj) = v.as_object_mut() {
            out.extend(std::mem::take(obj));
        }
        Value::Object(out)
    }
}

/// What the journal knows about the tool calls of one session.
#[derive(Debug, Default)]
pub struct ActionLog {
    started: HashSet<String>,
    finished: HashSet<String>,
}

impl ActionLog {
    pub fn from_records(records: &[ActionRecord]) -> Self {
        let mut log = Self::default();
        for r in records.iter().filter(|r| r.kind == ActionKind::Tool) {
            match r.phase {
                ActionPhase::Started => {
                    log.started.insert(r.id.clone());
                }
                p if p.is_finished() => {
                    log.finished.insert(r.id.clone());
                }
                _ => {}
            }
        }
        log
    }
    pub fn started(&self, id: &str) -> bool {
        self.started.contains(id)
    }
    pub fn finished(&self, id: &str) -> bool {
        self.finished.contains(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_round_trip() {
        let r = ActionRecord::new(ActionKind::Tool, "c1", "bash", ActionPhase::Started)
            .with_capability(Some(Capability::of(&crate::tools::bash::BashTool)));
        let line = r.to_line();
        assert_eq!(line["type"], "action");
        assert_eq!(line["phase"], "started");
        assert_eq!(line["effect"], "exec");
        assert_eq!(line["provider"], "tool");
        let mut obj = line.as_object().unwrap().clone();
        obj.remove("type");
        let back: ActionRecord = serde_json::from_value(Value::Object(obj)).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn log_tracks_tool_phases_only() {
        let recs = [
            ActionRecord::new(ActionKind::Tool, "a", "bash", ActionPhase::Started),
            ActionRecord::new(ActionKind::Tool, "a", "bash", ActionPhase::Failed),
            ActionRecord::new(ActionKind::Tool, "b", "edit", ActionPhase::Started),
            ActionRecord::new(ActionKind::Verify, "v", "cargo test", ActionPhase::Started),
        ];
        let log = ActionLog::from_records(&recs);
        assert!(log.started("a") && log.finished("a"));
        assert!(log.started("b") && !log.finished("b"));
        assert!(!log.started("v"));
    }
}
