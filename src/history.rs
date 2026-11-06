//! Keeps the "every tool call has exactly one result" invariant, and repairs
//! history after a crash.

use crate::journal::ActionLog;
use crate::models::{Message, Role, ToolCall};

pub const INTERRUPTED_CONTENT: &str = "[session restored: tool call interrupted]";
pub const NOT_STARTED_CONTENT: &str =
    "[session restored: tool call interrupted before it started; it did not run]";
pub const PARTIAL_CONTENT: &str = "[session restored: tool call interrupted while running; it may have partially run. Check its effects before retrying.]";
pub const LOST_RESULT_CONTENT: &str = "[session restored: tool call ran to the end but its result was lost. Check its effects instead of re-running it.]";

pub struct Recovery {
    pub messages: Vec<Message>,
    /// Read-only calls whose placeholder should be replaced by a re-run.
    pub rerun: Vec<ToolCall>,
    /// Whether any placeholder was inserted (the file must be rewritten).
    pub repaired: bool,
}

/// Drop the leading system message and insert placeholder results for
/// dangling tool calls. `journal` = what the execution journal recorded
/// (`None` for legacy sessions that journaled nothing).
pub fn recover(
    messages: Vec<Message>,
    journal: Option<&ActionLog>,
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
                    match journal {
                        None => INTERRUPTED_CONTENT,
                        Some(j) if j.finished(&tc.id) => LOST_RESULT_CONTENT,
                        Some(j) if j.started(&tc.id) => PARTIAL_CONTENT,
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
            ToolCall::new("e", "edit", json!({})),
        ]);
        use crate::journal::{ActionKind::Tool, ActionPhase::*, ActionRecord};
        let log = ActionLog::from_records(&[
            ActionRecord::new(Tool, "b", "bash", Started),
            ActionRecord::new(Tool, "e", "edit", Started),
            ActionRecord::new(Tool, "e", "edit", Completed),
        ]);
        let rec = recover(
            vec![Message::system("s"), Message::user("u"), a],
            Some(&log),
            |n| n == "read",
        );
        assert_eq!(rec.messages.len(), 6);
        assert_eq!(rec.messages[2].text(), INTERRUPTED_CONTENT);
        assert_eq!(rec.messages[3].text(), PARTIAL_CONTENT);
        assert_eq!(rec.messages[4].text(), NOT_STARTED_CONTENT);
        assert_eq!(rec.messages[5].text(), LOST_RESULT_CONTENT);
        assert_eq!(rec.rerun.len(), 1);
        assert!(rec.repaired);
    }
}
