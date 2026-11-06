//! Effect semantics: what running an action does to the world, and how a
//! crashed run of it is recovered.
//!
//! Every tool declares one [`Effect`]. Plan mode, subagent profiles and the
//! session journal all read it instead of asking each tool a yes/no
//! question of their own, and crash recovery picks its [`Recovery`]
//! strategy from it.
//!
//! 🦀 Rust notes
//! - `#[derive(Clone, Copy)]` on a fieldless enum makes it a plain value
//!   (like an integer): passing it around copies it, no borrowing needed.
//! - `#[serde(rename_all = "snake_case")]` writes `Effect::Fetch` as
//!   `"fetch"` in the session file.

use serde::{Deserialize, Serialize};

/// What an action does when it runs, from harmless to arbitrary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Reads local state only (read, grep, find, ls, session_search).
    Read,
    /// Reads from the network; nothing outside changes (web_fetch, web_search).
    Fetch,
    /// Changes only this session's own live state: the task board, a
    /// question to the user, a `run_code` transport whose sub-calls are
    /// gated one by one.
    Session,
    /// Writes local files, with the same result when repeated (write).
    Write,
    /// Changes local files or stores in ways a repeat may not undo
    /// (edit, memory, skill_manage).
    Mutate,
    /// Runs arbitrary commands: anything may happen (bash).
    Exec,
}

/// What resume does with a call that was interrupted by a crash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recovery {
    /// Run it again: repeating it is harmless.
    Replay,
    /// Do not re-run; the model is told it may have partially run and
    /// checks its effects first.
    Inspect,
}

impl Effect {
    /// Whether the action can change files or state outside the session
    /// (plan mode and read-only profiles block these).
    pub fn mutates(self) -> bool {
        matches!(self, Effect::Write | Effect::Mutate | Effect::Exec)
    }

    /// The recovery an action with this effect gets unless its tool says
    /// otherwise: reads replay, everything else is inspected.
    pub fn default_recovery(self) -> Recovery {
        match self {
            Effect::Read | Effect::Fetch => Recovery::Replay,
            _ => Recovery::Inspect,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_reads_replay_by_default() {
        for e in [Effect::Read, Effect::Fetch] {
            assert_eq!(e.default_recovery(), Recovery::Replay);
            assert!(!e.mutates());
        }
        assert!(!Effect::Session.mutates());
        assert_eq!(Effect::Session.default_recovery(), Recovery::Inspect);
        for e in [Effect::Write, Effect::Mutate, Effect::Exec] {
            assert!(e.mutates());
            assert_eq!(e.default_recovery(), Recovery::Inspect);
        }
    }

    #[test]
    fn serializes_snake_case() {
        assert_eq!(serde_json::to_value(Effect::Fetch).unwrap(), "fetch");
        assert_eq!(
            serde_json::from_value::<Recovery>("replay".into()).unwrap(),
            Recovery::Replay
        );
    }
}
