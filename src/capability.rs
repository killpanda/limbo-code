//! Capabilities: what a session can do, and who provides each one.
//!
//! The model still calls tools by name (its tool list does not change, so
//! the prompt cache stays intact). Underneath, every registered tool is a
//! [`Capability`]: a name, the [`Effect`] it has, how a crashed run is
//! recovered, and the [`Provider`] that carries it out. Admission and the
//! execution journal work in these terms, so a capability served by a
//! subagent or by the user is gated and recorded like any local tool.
//!
//! 🦀 Rust notes
//! - `Capability` is a plain value snapshot (`Copy`): reading it out of
//!   the registry does not keep the tool borrowed.

use serde::{Deserialize, Serialize};

use crate::effect::{Effect, Recovery};
use crate::tools::Tool;

/// Who carries a capability out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    /// A local tool run in this process (read, edit, bash, …).
    #[default]
    Tool,
    /// A program the model writes (`run_code`): its sub-calls are
    /// capabilities of their own, admitted one by one.
    Program,
    /// The task board and the subagents behind it (plan_tasks, task_*,
    /// and a child's checkpoint / need / ask_main / finish).
    Subagent,
    /// The user: the answer arrives as their next message (ask_user).
    Human,
    /// The harness itself, not a model call: `/goal` verify commands.
    Runtime,
    /// The chat the session is attached to (`limbo gateway`): `send_file`
    /// delivers to the person on Telegram / WeChat.
    Channel,
}

impl Provider {
    /// The word shown in tool cards, the trace and the journal.
    pub fn label(self) -> &'static str {
        match self {
            Provider::Tool => "tool",
            Provider::Program => "program",
            Provider::Subagent => "subagent",
            Provider::Human => "human",
            Provider::Runtime => "runtime",
            Provider::Channel => "channel",
        }
    }
}

/// One capability as admission and the journal see it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capability {
    pub name: &'static str,
    pub effect: Effect,
    pub recovery: Recovery,
    pub provider: Provider,
}

impl Capability {
    pub fn of(tool: &dyn Tool) -> Self {
        Self {
            name: tool.name(),
            effect: tool.effect(),
            recovery: tool.recovery(),
            provider: tool.provider(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolRegistry;

    #[test]
    fn core_catalog_providers() {
        let reg = ToolRegistry::core(true);
        let caps = reg.capabilities();
        let cap = |n: &str| *caps.iter().find(|c| c.name == n).unwrap();
        assert_eq!(cap("read").provider, Provider::Tool);
        assert_eq!(cap("read").recovery, Recovery::Replay);
        assert_eq!(cap("run_code").provider, Provider::Program);
        assert_eq!(cap("bash").effect, Effect::Exec);
    }

    #[test]
    fn labels_match_serde() {
        for p in [
            Provider::Tool,
            Provider::Program,
            Provider::Subagent,
            Provider::Human,
            Provider::Runtime,
            Provider::Channel,
        ] {
            assert_eq!(serde_json::to_value(p).unwrap(), p.label());
        }
    }

    #[test]
    fn ask_user_is_human() {
        let c = Capability::of(&crate::tools::ask_user::AskUserTool);
        assert_eq!(c.provider, Provider::Human);
        assert!(!c.effect.mutates());
    }
}
