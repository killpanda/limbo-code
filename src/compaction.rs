//! Context compaction: when the next request would overflow the window,
//! summarize the older part of the conversation into one message.
//!
//! Compaction is non-destructive on disk: a `compaction` record is appended
//! and the summarized messages stay in the file.

use crate::llm::usage::estimate_tokens;
use crate::models::{Message, Role};

pub const SUMMARY_TAG: &str = "conversation-summary";
const SERIALIZE_CAP: usize = 10_000;
pub const SUMMARY_SECTIONS: [&str; 6] = [
    "Goal",
    "Progress",
    "Key Decisions",
    "Files Touched",
    "Errors & Fixes",
    "Next Steps",
];

pub fn should_compact(
    estimated_prompt: u64,
    context_window: u64,
    enabled: bool,
    reserve: u64,
) -> bool {
    enabled && estimated_prompt > context_window.saturating_sub(reserve)
}

/// Index where the kept region starts, or `None` when there is nothing to
/// cut. Never starts the kept region with a dangling tool result.
pub fn find_split_point(messages: &[Message], keep_recent_tokens: u64) -> Option<usize> {
    let mut total = 0;
    let mut idx = messages.len();
    while idx > 1 {
        let cost = estimate_tokens(&messages[idx - 1..idx]);
        if total + cost > keep_recent_tokens {
            break;
        }
        total += cost;
        idx -= 1;
    }
    while idx < messages.len() && messages[idx].role == Role::Tool {
        idx += 1;
    }
    (idx > 1 && idx < messages.len()).then_some(idx)
}

fn serialize(m: &Message) -> Option<String> {
    let mut parts = Vec::new();
    if !m.text().is_empty() {
        let mut c: String = m.text().chars().take(SERIALIZE_CAP).collect();
        if m.text().chars().count() > SERIALIZE_CAP {
            c.push_str("\n[... truncated ...]");
        }
        parts.push(c);
    }
    if let Some(calls) = &m.tool_calls {
        let names: Vec<&str> = calls.iter().map(|c| c.name()).collect();
        parts.push(format!("[tool calls: {}]", names.join(", ")));
    }
    if parts.is_empty() {
        return None;
    }
    let role = match m.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };
    Some(format!("{role}: {}", parts.join("\n")))
}

pub fn build_summary_prompt(to_summarize: &[Message], previous: Option<&str>) -> Vec<Message> {
    let rendered: Vec<String> = to_summarize.iter().filter_map(serialize).collect();
    let sections: Vec<String> = SUMMARY_SECTIONS.iter().map(|s| format!("## {s}")).collect();
    let mut prompt = format!(
        "You are compacting a coding-agent conversation that is about to overflow its context window. Summarize \
         the conversation excerpt below so work can continue without it.\n\nRequirements:\n- Write the summary \
         with EXACTLY these six section headers:\n{}\n- Keep the total under 2000 characters; be terse and \
         factual.\n- Preserve concrete file paths, command names, error messages, and decisions verbatim where \
         they matter.\n",
        sections.join("\n")
    );
    if let Some(prev) = previous.filter(|p| !p.is_empty()) {
        prompt.push_str(&format!(
            "\nAn earlier summary covers the conversation before this excerpt. Fold the excerpt into it: produce \
             ONE updated summary in the same six-section format, not a separate addendum.\n\n<previous-summary>\n\
             {prev}\n</previous-summary>\n"
        ));
    }
    prompt.push_str(&format!(
        "\n<conversation>\n{}\n</conversation>\n",
        rendered.join("\n\n")
    ));
    vec![Message::user(prompt)]
}

/// A summary as a user message (so resume keeps it).
pub fn make_summary_message(summary: &str) -> Message {
    Message::user(format!(
        "<{SUMMARY_TAG}>\nEarlier conversation was compacted into this summary. Treat it as established context; \
         the messages after it are the most recent part of the conversation.\n\n{}\n</{SUMMARY_TAG}>",
        summary.trim()
    ))
}

pub fn is_summary_message(m: &Message) -> bool {
    m.role == Role::User && m.text().starts_with(&format!("<{SUMMARY_TAG}>"))
}

/// The summary text inside a summary message.
pub fn summary_body(m: &Message) -> Option<String> {
    if !is_summary_message(m) {
        return None;
    }
    let t = m.text();
    let start = t.find("\n\n").map(|i| i + 2)?;
    let end = t.rfind(&format!("</{SUMMARY_TAG}>"))?;
    Some(t[start..end].trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_never_starts_with_tool_result() {
        let mut a = Message::assistant("");
        a.tool_calls = Some(vec![crate::models::ToolCall::new(
            "c",
            "ls",
            serde_json::json!({}),
        )]);
        let msgs = vec![
            Message::system("s"),
            Message::user("x".repeat(4000)),
            a,
            Message::tool("c", "out"),
            Message::user("y"),
        ];
        let idx = find_split_point(&msgs, 30).unwrap();
        assert_ne!(msgs[idx].role, Role::Tool);
    }

    #[test]
    fn summary_roundtrip() {
        let m = make_summary_message("## Goal\nship");
        assert!(is_summary_message(&m));
        assert_eq!(summary_body(&m).unwrap(), "## Goal\nship");
    }
}
