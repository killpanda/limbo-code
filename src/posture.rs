//! Posture: how a session works, as independent dimensions instead of a
//! growing list of modes.
//!
//! | dimension       | values              | slash command  |
//! |-----------------|---------------------|----------------|
//! | policy          | default, plan       | `/plan`        |
//! | capability set  | coding, daily       | `/daily-mode`  |
//! | presentation    | direct, code        | `/code-mode`   |
//! | strategy        | single, goal        | `/goal`        |
//!
//! Each dimension changes one thing: the policy decides what is admitted
//! (`admission.rs`), the capability set decides which tools and prompt the
//! session has, the presentation decides how the tools are shown to the
//! model (one by one, or behind `run_code`), and the strategy decides how
//! turns are driven (`pump.rs`). The slash commands are sugar over these.
//!
//! The session file keeps the old `plan_mode` / `daily_mode` / `code_mode`
//! fields, so files stay readable by older versions; [`Posture::from_meta`]
//! and [`Posture::write_meta`] translate.
//!
//! 🦀 Rust notes
//! - `#[derive(Default)]` with `#[default]` on a variant picks that variant
//!   for `Posture::default()`.

use crate::session::SessionMeta;

/// What is admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolicyProfile {
    #[default]
    Default,
    /// Read-only: nothing whose effect mutates runs (`PlanPolicy`).
    Plan,
}

/// Which tools and system prompt the session has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CapabilitySet {
    #[default]
    Coding,
    /// The general-assistant prompt plus the daily tools (web_fetch,
    /// memory, session_search, ask_user, skill_manage).
    Daily,
}

/// How the tools are presented to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Presentation {
    /// One function definition per tool.
    #[default]
    Direct,
    /// Code Mode: the catalog collapses to `run_code` and a Python SDK.
    Code,
}

/// How turns are driven.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// One user input, one turn.
    Single,
    /// `/goal`: repeat turns until the verify command passes.
    Goal,
}

/// The policy, capability set and presentation of a session. The strategy
/// is not stored here: it follows from whether a goal is live
/// ([`Posture::strategy`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Posture {
    pub policy: PolicyProfile,
    pub capabilities: CapabilitySet,
    pub presentation: Presentation,
}

impl Posture {
    pub fn plan(&self) -> bool {
        self.policy == PolicyProfile::Plan
    }
    pub fn daily(&self) -> bool {
        self.capabilities == CapabilitySet::Daily
    }
    pub fn code(&self) -> bool {
        self.presentation == Presentation::Code
    }

    pub fn with_plan(mut self, on: bool) -> Self {
        self.policy = if on {
            PolicyProfile::Plan
        } else {
            PolicyProfile::Default
        };
        self
    }
    pub fn with_daily(mut self, on: bool) -> Self {
        self.capabilities = if on {
            CapabilitySet::Daily
        } else {
            CapabilitySet::Coding
        };
        self
    }
    pub fn with_code(mut self, on: bool) -> Self {
        self.presentation = if on {
            Presentation::Code
        } else {
            Presentation::Direct
        };
        self
    }

    /// What a fresh session started from this one keeps (`/new`): the
    /// capability set and presentation, not the policy.
    pub fn carried_over(self) -> Self {
        self.with_plan(false)
    }

    pub fn from_meta(meta: &SessionMeta) -> Self {
        Self::default()
            .with_plan(meta.plan_mode)
            .with_daily(meta.daily_mode)
            .with_code(meta.code_mode)
    }

    pub fn write_meta(self, meta: &mut SessionMeta) {
        meta.plan_mode = self.plan();
        meta.daily_mode = self.daily();
        meta.code_mode = self.code();
    }

    /// The strategy a session with this meta runs.
    pub fn strategy(meta: &SessionMeta) -> Strategy {
        if meta.goal.as_ref().is_some_and(|g| g.is_live()) {
            Strategy::Goal
        } else {
            Strategy::Single
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn meta_round_trip() {
        let mut meta = SessionMeta::new("s".into(), Path::new("/w"), "m");
        let p = Posture::default().with_plan(true).with_code(true);
        p.write_meta(&mut meta);
        assert!(meta.plan_mode && meta.code_mode && !meta.daily_mode);
        assert_eq!(Posture::from_meta(&meta), p);
        assert_eq!(Posture::strategy(&meta), Strategy::Single);
    }

    #[test]
    fn new_session_drops_plan_only() {
        let p = Posture::default()
            .with_plan(true)
            .with_daily(true)
            .with_code(true);
        let q = p.carried_over();
        assert!(!q.plan() && q.daily() && q.code());
    }
}
