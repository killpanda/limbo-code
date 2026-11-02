//! Shared data models: messages, tool calls, tool results and LLM events.
//!
//! These types are (de)serialized with serde in exactly the same JSON shape
//! the Python version writes, so a session file created by either
//! implementation can be resumed by the other.
//!
//! 🦀 Rust notes
//! - `#[derive(Serialize, Deserialize)]` asks serde to generate the JSON code
//!   for us at compile time — no reflection at runtime.
//! - `Option<T>` is how Rust says "may be missing". Combined with
//!   `skip_serializing_if = "Option::is_none"` a `None` field is simply not
//!   written, which keeps the JSONL compact.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Make a random 16-hex-char id (same format as Python's `secrets.token_hex(8)`).
pub fn new_id() -> String {
    let bytes: [u8; 8] = rand::random();
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A file or image attached to a message. Only the path is persisted, never
/// the bytes, so a restored session may point at a file that is gone.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Attachment {
    /// `"image"` or `"file"`.
    pub kind: String,
    pub name: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
}

/// The `function` half of a stored tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// Kept as a JSON object (Python stores a dict); clients serialize it to
    /// a string only when the wire format demands it.
    #[serde(default)]
    pub arguments: Value,
}

/// A tool call as stored on an assistant message (OpenAI shape).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            kind: function_type(),
            function: FunctionCall {
                name: name.into(),
                arguments,
            },
        }
    }
    pub fn name(&self) -> &str {
        &self.function.name
    }
}

/// Who wrote a message.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One chat message — one line of the session JSONL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// Addressable entry id (compaction records point at the first kept id).
    #[serde(default = "new_id")]
    pub id: String,
    pub role: Role,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub attachments: Option<Vec<Attachment>>,
    /// Reasoning text from thinking models, replayed when an API requires it.
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Signature for replayed thinking (Anthropic) or a serialized
    /// Responses reasoning item (prefixed `responses:`).
    #[serde(default)]
    pub reasoning_signature: Option<String>,
}

impl Message {
    fn base(role: Role, content: Option<String>) -> Self {
        Self {
            id: new_id(),
            role,
            content,
            tool_calls: None,
            tool_call_id: None,
            attachments: None,
            reasoning: None,
            reasoning_signature: None,
        }
    }
    pub fn system(text: impl Into<String>) -> Self {
        Self::base(Role::System, Some(text.into()))
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self::base(Role::User, Some(text.into()))
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::base(Role::Assistant, Some(text.into()))
    }
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        let mut m = Self::base(Role::Tool, Some(content.into()));
        m.tool_call_id = Some(call_id.into());
        m
    }
    /// The text content, or `""`.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
    }
}

/// What a tool returns.
#[derive(Debug, Clone, Default)]
pub struct ToolResult {
    pub success: bool,
    pub output: Option<String>,
    pub error: Option<String>,
    pub attachments: Option<Vec<Attachment>>,
    /// The tool asks the agent to end the turn right after this batch
    /// (`ask_user`): the user's next message is the answer.
    pub end_turn: bool,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: Some(output.into()),
            ..Default::default()
        }
    }
    pub fn err(error: impl Into<String>) -> Self {
        Self {
            success: false,
            error: Some(error.into()),
            ..Default::default()
        }
    }
    /// The text the model sees for this result (output, else error).
    pub fn content_for_model(&self) -> String {
        self.output
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| self.error.clone())
            .unwrap_or_default()
    }
}

/// Response-level metadata emitted once per LLM call, after all chunks.
#[derive(Debug, Clone, Default)]
pub struct CompletionMeta {
    pub usage: Option<Value>,
    pub finish_reason: Option<String>,
    pub ttft: Option<f64>,
    pub duration: Option<f64>,
}

/// Streaming events a provider client produces.
///
/// 🦀 A Rust `enum` can carry data in each variant — perfect for "one of
/// these kinds of event". `match` then forces us to handle every case.
#[derive(Debug, Clone)]
pub enum LlmEvent {
    Text(String),
    Thinking {
        text: String,
        signature: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    Done(CompletionMeta),
}
