//! Chat channels: the platform half of the chat gateway (`limbo gateway`).
//!
//! A [`Channel`] adapter only speaks one platform's protocol — it long-polls
//! for messages, sends text / files / typing indicators and downloads media.
//! Everything else (who may talk, which session a chat maps to, how agent
//! events become chat messages) lives in `gateway/` and never sees a
//! Telegram or WeChat field name, the same way `llm/usage.rs` hides each
//! provider's usage fields from the agent loop.
//!
//! Two adapters exist: [`telegram`] (official Bot API) and [`weixin`]
//! (personal WeChat through Tencent's iLink bot API). [`mock`] is an
//! in-memory adapter for tests.
//!
//! 🦀 Rust notes
//! - `#[async_trait]` lets a trait have `async fn`s that work behind
//!   `Arc<dyn Channel>` (a trait object); the gateway keeps one such
//!   object per configured channel.
//! - `thiserror` derives `std::error::Error` + `Display` for [`SendError`]
//!   from the `#[error(..)]` attributes.

pub mod markdown;
#[cfg(test)]
pub mod mock;
pub mod store;
pub mod telegram;
pub mod weixin;

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Which platform a chat lives on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum ChannelKind {
    Telegram,
    Weixin,
    #[cfg(test)]
    Mock,
}

impl ChannelKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ChannelKind::Telegram => "telegram",
            ChannelKind::Weixin => "weixin",
            #[cfg(test)]
            ChannelKind::Mock => "mock",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "telegram" | "tg" => Some(ChannelKind::Telegram),
            "weixin" | "wechat" | "wx" => Some(ChannelKind::Weixin),
            _ => None,
        }
    }
}

/// One chat, globally unique: which channel, which bot account, which chat
/// (and which forum topic, for Telegram topic groups).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct PeerKey {
    pub channel: ChannelKind,
    pub account: String,
    pub chat: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
}

impl PeerKey {
    /// `telegram:<bot>:<chat>[:<topic>]` — the key in `sessions.json`.
    pub fn key(&self) -> String {
        let mut k = format!("{}:{}:{}", self.channel.as_str(), self.account, self.chat);
        if let Some(t) = &self.thread {
            k.push(':');
            k.push_str(t);
        }
        k
    }
}

/// What kind of media a message carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    File,
    Voice,
    Video,
}

/// A media item of an inbound message — only a reference; the gateway
/// downloads it with [`Channel::fetch_media`] after admission.
#[derive(Debug, Clone)]
pub struct MediaRef {
    pub kind: MediaKind,
    /// The file name, when the platform tells it.
    pub name: Option<String>,
    pub size: Option<u64>,
    /// Adapter-private data (Telegram `file_id`, WeChat CDN parameters).
    pub data: serde_json::Value,
}

/// Opaque per-message reply context. WeChat needs the inbound message's
/// `context_token` to answer; Telegram needs nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplyCtx(pub Option<String>);

/// One message from a chat.
#[derive(Debug, Clone)]
pub struct InboundMessage {
    pub peer: PeerKey,
    /// Stable platform user id of the sender (whitelist key).
    pub sender: String,
    /// A display name for logs.
    pub sender_name: String,
    /// Platform message id (deduplication).
    pub msg_id: String,
    pub text: String,
    pub media: Vec<MediaRef>,
    pub reply_ctx: ReplyCtx,
    /// Sent in a group / channel (not a private chat).
    pub is_group: bool,
    /// The message addresses the bot (mention, reply to it, private chat).
    pub addressed: bool,
}

/// How a text message is formatted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextFormat {
    /// Send verbatim.
    Plain,
    /// Model markdown: converted to the channel's dialect by the adapter.
    Markdown,
}

/// A platform message the gateway may edit later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MsgHandle(pub String);

/// Something to send to a chat.
#[derive(Debug, Clone)]
pub enum OutboundMessage {
    Text {
        text: String,
        format: TextFormat,
    },
    /// Replace the text of an earlier message (only when `caps.edit`).
    Edit {
        handle: MsgHandle,
        text: String,
        format: TextFormat,
    },
    File {
        path: PathBuf,
        caption: Option<String>,
    },
    Typing,
    /// A question with options (`ask_user`); buttons when `caps.buttons`.
    Choices {
        question: String,
        options: Vec<String>,
    },
}

/// The markdown dialect a channel renders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkdownDialect {
    /// Telegram's HTML subset (`parse_mode = HTML`).
    TelegramHtml,
    /// No formatting at all (WeChat).
    Plain,
}

/// What a channel can do; the renderer adapts to it.
#[derive(Debug, Clone)]
pub struct ChannelCaps {
    /// Hard limit of one text message, in characters.
    pub max_text_chars: usize,
    /// Messages can be edited (streaming answers by editing one message).
    pub edit: bool,
    pub markdown: MarkdownDialect,
    /// Minimum gap between two sends / edits to the same chat.
    pub min_send_interval: Duration,
    /// How long a typing indicator lasts before it must be re-sent.
    pub typing_ttl: Duration,
}

/// Why a send failed — the outbox decides from this whether to retry.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    #[error("rate limited, retry after {0:?}")]
    RateLimited(Duration),
    /// WeChat: the reply window for this chat is closed; resend after the
    /// user's next message.
    #[error("reply context expired")]
    ContextExpired,
    /// The bot token is no longer valid (log in again).
    #[error("authorization expired: {0}")]
    AuthExpired(String),
    /// Network trouble or a 5xx: worth a retry.
    #[error("transient: {0}")]
    Transient(String),
    #[error("{0}")]
    Other(String),
}

/// A platform adapter.
#[async_trait]
pub trait Channel: Send + Sync {
    fn kind(&self) -> ChannelKind;
    /// The bot account this adapter runs as (part of every [`PeerKey`]).
    fn account(&self) -> String;
    fn caps(&self) -> ChannelCaps;
    /// Long-poll until `cancel`; push every message into `inbox`. Persists
    /// its own cursor after the message was handed over.
    async fn run(
        &self,
        inbox: mpsc::Sender<InboundMessage>,
        cancel: CancellationToken,
    ) -> anyhow::Result<()>;
    /// Send one message; `Some(handle)` for messages that can be edited.
    async fn send(
        &self,
        peer: &PeerKey,
        ctx: &ReplyCtx,
        msg: OutboundMessage,
    ) -> Result<Option<MsgHandle>, SendError>;
    /// Download one media item into `dir`; returns the file written.
    async fn fetch_media(
        &self,
        media: &MediaRef,
        dir: &Path,
        max_bytes: u64,
    ) -> anyhow::Result<PathBuf>;
}

/// A file name that is safe to create inside a media directory.
pub fn safe_file_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_start_matches('.');
    if trimmed.is_empty() {
        "file".into()
    } else {
        trimmed.chars().take(120).collect()
    }
}

/// Build a multipart/form-data body by hand (reqwest's `multipart` feature
/// is not enabled): text fields, then one file field.
pub fn multipart_body(
    fields: &[(&str, String)],
    file_field: &str,
    file_name: &str,
    content: &[u8],
) -> (String, Vec<u8>) {
    let boundary = format!("limbo-{}", crate::agent::new_session_id().replace('-', ""));
    let mut body = Vec::with_capacity(content.len() + 512);
    for (k, v) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
                .as_bytes(),
        );
    }
    let fname = file_name.replace(['"', '\r', '\n'], "_");
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{fname}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(content);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_key_strings() {
        let p = PeerKey {
            channel: ChannelKind::Telegram,
            account: "42".into(),
            chat: "-100".into(),
            thread: Some("7".into()),
        };
        assert_eq!(p.key(), "telegram:42:-100:7");
    }

    #[test]
    fn safe_names() {
        assert_eq!(safe_file_name("../../etc/passwd"), "_.._etc_passwd");
        assert_eq!(safe_file_name("报告 v2.pdf"), "报告_v2.pdf");
        assert_eq!(safe_file_name(""), "file");
    }

    #[test]
    fn multipart_has_fields_and_file() {
        let (ct, body) = multipart_body(&[("chat_id", "1".into())], "document", "a.txt", b"hi");
        let b = String::from_utf8(body).unwrap();
        let boundary = ct.split("boundary=").nth(1).unwrap();
        assert!(b.starts_with(&format!("--{boundary}\r\n")));
        assert!(b.contains("name=\"chat_id\"\r\n\r\n1\r\n"));
        assert!(b.contains("filename=\"a.txt\""));
        assert!(b.ends_with(&format!("\r\n--{boundary}--\r\n")));
    }
}
