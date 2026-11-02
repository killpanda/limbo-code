//! Keeps the "every tool call has exactly one result" invariant, and repairs
//! history after a crash.

use std::collections::HashSet;

use crate::models::{Message, Role, ToolCall};

pub const INTERRUPTED_CONTENT: &str = "[session restored: tool call interrupted]";
pub const NOT_STARTED_CONTENT: &str =
    "[session restored: tool call interrupted before it started; it did not run]";
pub const PARTIAL_CONTENT: &str = "[session restored: tool call interrupted while running; it may have partially run. Check its effects before retrying.]";

pub struct Recovery {
    pub messages: Vec<Message>,
    /// Read-only calls whose placeholder should be replaced by a re-run.
    pub rerun: Vec<ToolCall>,
    /// Whether any placeholder was inserted (the file must be rewritten).
    pub repaired: bool,
}

/// Drop the leading system message and insert placeholder results for
/// dangling tool calls. `started` = calls whose intent was journaled
/// (`None` for legacy sessions without intents).
pub fn recover(
    messages: Vec<Message>,
    started: Option<&HashSet<String>>,
    is_safe: impl Fn(&str) -> bool,
) -> Recovery {
    let mut history = messages;
    if history.first().is_some_and(|m| m.role == Role::System) {
        history.remove(0);
    }
    let mut out = Vec::with_capacity(history.len());
    let mut rerun = Vec::new();
    let mut pending: Vec<ToolCall> = Vec::new();
    let mut repaired = false;

    let mut flush =
        |out: &mut Vec<Message>, pending: &mut Vec<ToolCall>, rerun: &mut Vec<ToolCall>| {
            for tc in pending.drain(..) {
                repaired = true;
                let content = if is_safe(tc.name()) {
                    rerun.push(tc.clone());
                    INTERRUPTED_CONTENT
                } else {
                    match started {
                        None => INTERRUPTED_CONTENT,
                        Some(s) if s.contains(&tc.id) => PARTIAL_CONTENT,
                        Some(_) => NOT_STARTED_CONTENT,
                    }
                };
                out.push(Message::tool(tc.id.clone(), content));
            }
        };

    for m in history {
        match m.role {
            Role::Assistant if m.has_tool_calls() => {
                flush(&mut out, &mut pending, &mut rerun);
                pending.extend(
                    m.tool_calls
                        .iter()
                        .flatten()
                        .filter(|c| !c.id.is_empty())
                        .cloned(),
                );
            }
            Role::Tool => pending.retain(|c| Some(&c.id) != m.tool_call_id.as_ref()),
            _ => flush(&mut out, &mut pending, &mut rerun),
        }
        out.push(m);
    }
    flush(&mut out, &mut pending, &mut rerun);
    Recovery {
        messages: out,
        rerun,
        repaired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn placeholders_follow_intents() {
        let mut a = Message::assistant("");
        a.tool_calls = Some(vec![
            ToolCall::new("r", "read", json!({})),
            ToolCall::new("b", "bash", json!({})),
            ToolCall::new("w", "write", json!({})),
        ]);
        let started: HashSet<String> = ["b".to_string()].into();
        let rec = recover(
            vec![Message::system("s"), Message::user("u"), a],
            Some(&started),
            |n| n == "read",
        );
        assert_eq!(rec.messages.len(), 5);
        assert_eq!(rec.messages[2].text(), INTERRUPTED_CONTENT);
        assert_eq!(rec.messages[3].text(), PARTIAL_CONTENT);
        assert_eq!(rec.messages[4].text(), NOT_STARTED_CONTENT);
        assert_eq!(rec.rerun.len(), 1);
        assert!(rec.repaired);
    }
}
