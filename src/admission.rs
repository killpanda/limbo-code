//! Admission: the one gate every action passes before it runs.
//!
//! A [`Policy`] looks at an [`ActionRequest`] (which tool, its arguments,
//! its declared [`Effect`]) and admits or denies it. Plan mode is a policy
//! ([`PlanPolicy`]), and so is every subagent profile restriction
//! (`subagents/policies.rs`). The tool registry runs the policies in order
//! and the first denial wins; the denial text goes back to the model as an
//! error result it can read and react to.
//!
//! 🦀 Rust notes
//! - `ActionRequest<'a>` borrows the tool and its arguments for the
//!   duration of the check (`'a` is that borrow's lifetime): nothing is
//!   cloned just to ask a question.
//! - `Policy: Send + Sync` lets one policy object be shared by calls
//!   running concurrently on different tasks.

use serde_json::Value;

use crate::effect::Effect;
use crate::tools::Tool;

/// One proposed action, as the policies see it.
pub struct ActionRequest<'a> {
    pub tool: &'a dyn Tool,
    pub args: &'a Value,
}

impl ActionRequest<'_> {
    pub fn name(&self) -> &'static str {
        self.tool.name()
    }
    pub fn effect(&self) -> Effect {
        self.tool.effect()
    }
}

/// A policy's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Allow,
    /// Blocked; the message is what the model reads.
    Deny(String),
}

/// A rule in front of every action.
pub trait Policy: Send + Sync {
    fn admit(&self, req: &ActionRequest) -> Verdict;
    /// Whether the tool is listed to the model at all (hidden tools are
    /// still checked by [`Policy::admit`] when called).
    fn visible(&self, _tool: &dyn Tool) -> bool {
        true
    }
}

/// Run `policies` in order; the first denial wins.
pub fn admit<'p>(
    policies: impl IntoIterator<Item = &'p dyn Policy>,
    req: &ActionRequest,
) -> Verdict {
    for p in policies {
        if let v @ Verdict::Deny(_) = p.admit(req) {
            return v;
        }
    }
    Verdict::Allow
}

/// `/plan`: nothing that mutates runs or is listed.
pub struct PlanPolicy;

impl Policy for PlanPolicy {
    fn admit(&self, req: &ActionRequest) -> Verdict {
        if !req.effect().mutates() {
            return Verdict::Allow;
        }
        Verdict::Deny(format!(
            "Plan mode is on (read-only): {} is blocked. Keep exploring with read-only tools, then \
             present the plan; the user leaves plan mode with /plan off before anything is changed.",
            req.name()
        ))
    }
    fn visible(&self, tool: &dyn Tool) -> bool {
        !tool.mutates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{bash::BashTool, read::ReadTool};
    use serde_json::json;

    struct DenyAll(&'static str);
    impl Policy for DenyAll {
        fn admit(&self, _req: &ActionRequest) -> Verdict {
            Verdict::Deny(self.0.into())
        }
    }

    #[test]
    fn first_denial_wins() {
        let args = json!({});
        let req = ActionRequest {
            tool: &BashTool,
            args: &args,
        };
        let (a, b) = (DenyAll("a"), DenyAll("b"));
        let ps: [&dyn Policy; 3] = [&PlanPolicy, &a, &b];
        assert!(matches!(admit(ps, &req), Verdict::Deny(m) if m.contains("Plan mode")));
        let ps: [&dyn Policy; 2] = [&a, &b];
        assert_eq!(admit(ps, &req), Verdict::Deny("a".into()));
    }

    #[test]
    fn plan_admits_reads_only() {
        let args = json!({});
        let read = ActionRequest {
            tool: &ReadTool,
            args: &args,
        };
        assert_eq!(PlanPolicy.admit(&read), Verdict::Allow);
        assert!(PlanPolicy.visible(&ReadTool));
        assert!(!PlanPolicy.visible(&BashTool));
    }
}
