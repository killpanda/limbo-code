//! Personal WeChat through Tencent's official iLink bot channel (the
//! protocol behind the "微信 ClawBot" plugin): QR-code login, long-polled
//! `getupdates`, replies that carry the inbound message's `context_token`,
//! and AES-encrypted CDN media.
//!
//! Limits of the channel, not of this adapter: private chats only (group
//! messages are dropped), no message editing, and replies are only possible
//! for a while after the user's last message — a reply that comes too late
//! fails with [`SendError::ContextExpired`] and the gateway parks it until
//! the user writes again.
//!
//! The channel was built for OpenClaw; third-party clients use it at their
//! own risk (see the design doc). Requests identify themselves honestly as
//! `limbo/<version>` in `base_info.bot_agent`.

pub mod api;
pub mod cdn;
pub mod login;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{
    Channel, ChannelCaps, ChannelKind, InboundMessage, MarkdownDialect, MediaKind, MediaRef,
    MsgHandle, OutboundMessage, PeerKey, ReplyCtx, SendError, TextFormat, markdown, store,
};
use api::{Api, ITEM_FILE, ITEM_IMAGE, ITEM_TEXT, ITEM_VIDEO, ITEM_VOICE, MESSAGE_TYPE_USER};

pub struct Weixin {
    api: Api,
    bot_id: String,
    /// `typing_ticket` per user (from `getconfig`).
    tickets: Mutex<HashMap<String, String>>,
    /// The freshest `context_token` per user: replies use it when the
    /// gateway has none of its own (e.g. a message parked across restarts).
    contexts: Mutex<HashMap<String, String>>,
}

impl Weixin {
    pub fn new(
        http: reqwest::Client,
        base_url: &str,
        token: &str,
        bot_id: &str,
        id: api::Identity,
    ) -> Self {
        Self {
            api: Api {
                http,
                base_url: if base_url.is_empty() {
                    api::DEFAULT_BASE_URL.to_string()
                } else {
                    base_url.to_string()
                },
                token: token.to_string(),
                id,
            },
            bot_id: bot_id.to_string(),
            tickets: Mutex::new(HashMap::new()),
            contexts: Mutex::new(HashMap::new()),
        }
    }

    fn cursor_name(&self) -> String {
        format!("weixin-{}", self.bot_id)
    }

    /// One `WeixinMessage` → an inbound message (None: not a user message).
    pub fn parse_message(&self, m: &Value) -> Option<InboundMessage> {
        if m.get("message_type").and_then(Value::as_i64) != Some(MESSAGE_TYPE_USER) {
            return None;
        }
        let from = m.get("from_user_id").and_then(Value::as_str)?.to_string();
        let group = m
            .get("group_id")
            .and_then(Value::as_str)
            .filter(|g| !g.is_empty());
        let items = m
            .get("item_list")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut texts: Vec<String> = Vec::new();
        let mut media = Vec::new();
        for it in &items {
            if let Some(q) = it
                .pointer("/ref_msg/message_item/text_item/text")
                .and_then(Value::as_str)
                .filter(|q| !q.trim().is_empty())
            {
                let quoted: Vec<String> = q.lines().map(|l| format!("> {l}")).collect();
                texts.push(quoted.join("\n"));
            }
            match it.get("type").and_then(Value::as_i64).unwrap_or(0) {
                ITEM_TEXT => {
                    if let Some(t) = it.pointer("/text_item/text").and_then(Value::as_str) {
                        texts.push(t.to_string());
                    }
                }
                ITEM_VOICE => {
                    // The server's speech-to-text, when it has one.
                    match it.pointer("/voice_item/text").and_then(Value::as_str) {
                        Some(t) if !t.trim().is_empty() => texts.push(t.to_string()),
                        _ => media.push(MediaRef {
                            kind: MediaKind::Voice,
                            name: Some("voice.silk".into()),
                            size: None,
                            data: it.get("voice_item").cloned().unwrap_or(Value::Null),
                        }),
                    }
                }
                ITEM_IMAGE => media.push(MediaRef {
                    kind: MediaKind::Image,
                    name: Some("image.jpg".into()),
                    size: it.pointer("/image_item/mid_size").and_then(Value::as_u64),
                    data: it.get("image_item").cloned().unwrap_or(Value::Null),
                }),
                ITEM_FILE => media.push(MediaRef {
                    kind: MediaKind::File,
                    name: it
                        .pointer("/file_item/file_name")
                        .and_then(Value::as_str)
                        .map(String::from),
                    size: it
                        .pointer("/file_item/len")
                        .and_then(|v| v.as_str().and_then(|s| s.parse().ok()).or(v.as_u64())),
                    data: it.get("file_item").cloned().unwrap_or(Value::Null),
                }),
                ITEM_VIDEO => media.push(MediaRef {
                    kind: MediaKind::Video,
                    name: Some("video.mp4".into()),
                    size: it.pointer("/video_item/video_size").and_then(Value::as_u64),
                    data: it.get("video_item").cloned().unwrap_or(Value::Null),
                }),
                _ => {}
            }
        }
        let text = texts.join("\n").trim().to_string();
        if text.is_empty() && media.is_empty() {
            return None;
        }
        let ctx = m
            .get("context_token")
            .and_then(Value::as_str)
            .map(String::from);
        if let Some(c) = &ctx {
            self.contexts
                .lock()
                .unwrap()
                .insert(from.clone(), c.clone());
        }
        let id = m
            .get("message_id")
            .or_else(|| m.get("seq"))
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default();
        Some(InboundMessage {
            peer: PeerKey {
                channel: ChannelKind::Weixin,
                account: self.bot_id.clone(),
                chat: group.map(String::from).unwrap_or_else(|| from.clone()),
                thread: None,
            },
            sender: from,
            sender_name: String::new(),
            msg_id: id,
            text,
            media,
            reply_ctx: ReplyCtx(ctx),
            is_group: group.is_some(),
            // The iLink channel is private-chat only.
            addressed: group.is_none(),
        })
    }

    fn context_for(&self, peer: &PeerKey, ctx: &ReplyCtx) -> Option<String> {
        ctx.0
            .clone()
            .or_else(|| self.contexts.lock().unwrap().get(&peer.chat).cloned())
    }

    async fn typing(&self, user: &str, ctx: Option<&str>) -> Result<(), SendError> {
        let cached = self.tickets.lock().unwrap().get(user).cloned();
        let ticket = match cached {
            Some(t) => t,
            None => {
                let r = self.api.get_config(user, ctx).await?;
                let t = r
                    .get("typing_ticket")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if t.is_empty() {
                    return Ok(());
                }
                self.tickets
                    .lock()
                    .unwrap()
                    .insert(user.to_string(), t.clone());
                t
            }
        };
        self.api.send_typing(user, &ticket).await.map(|_| ())
    }

    async fn send_file(
        &self,
        to: &str,
        ctx: Option<&str>,
        path: &Path,
        caption: Option<&str>,
    ) -> Result<(), SendError> {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| SendError::Other(format!("cannot read {}: {e}", path.display())))?;
        if bytes.len() > 100 * 1024 * 1024 {
            return Err(SendError::Other("file larger than 100 MB".into()));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file".into());
        let is_image = crate::attachments::attachment_for(path).kind == "image";
        let media_type = if is_image {
            api::UPLOAD_IMAGE
        } else {
            api::UPLOAD_FILE
        };
        let up = cdn::upload(&self.api, to, media_type, &bytes)
            .await
            .map_err(|e| SendError::Transient(e.to_string()))?;
        if let Some(c) = caption.filter(|c| !c.is_empty()) {
            self.api
                .send_items(
                    to,
                    ctx,
                    vec![json!({"type": ITEM_TEXT, "text_item": {"text": c}})],
                )
                .await?;
        }
        let item = if is_image {
            json!({"type": ITEM_IMAGE, "image_item": {"media": up.media_json(), "mid_size": up.cipher_size}})
        } else {
            json!({"type": ITEM_FILE, "file_item": {"media": up.media_json(), "file_name": name, "len": up.raw_size.to_string()}})
        };
        self.api.send_items(to, ctx, vec![item]).await.map(|_| ())
    }
}

#[async_trait]
impl Channel for Weixin {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Weixin
    }
    fn account(&self) -> String {
        self.bot_id.clone()
    }
    fn caps(&self) -> ChannelCaps {
        ChannelCaps {
            max_text_chars: 2000,
            edit: false,
            markdown: MarkdownDialect::Plain,
            min_send_interval: Duration::from_millis(1500),
            typing_ttl: Duration::from_secs(8),
        }
    }

    async fn run(
        &self,
        inbox: mpsc::Sender<InboundMessage>,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        let cursor = self.cursor_name();
        let mut buf = store::read_cursor(&cursor).unwrap_or_default();
        let mut backoff = 1u64;
        loop {
            let res = tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                r = self.api.get_updates(&buf) => r,
            };
            let resp = match res {
                Ok(v) => {
                    backoff = 1;
                    v
                }
                Err(SendError::AuthExpired(d)) => {
                    anyhow::bail!("WeChat: {d}; run `limbo channels login weixin` again")
                }
                Err(e) => {
                    eprintln!("weixin: getupdates failed: {e}; retrying in {backoff}s");
                    tokio::select! {
                        _ = cancel.cancelled() => return Ok(()),
                        _ = tokio::time::sleep(Duration::from_secs(backoff)) => {}
                    }
                    backoff = (backoff * 2).min(60);
                    continue;
                }
            };
            for m in resp
                .get("msgs")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                if let Some(msg) = self.parse_message(&m)
                    && inbox.send(msg).await.is_err()
                {
                    return Ok(());
                }
            }
            if let Some(next) = resp.get("get_updates_buf").and_then(Value::as_str)
                && next != buf
            {
                buf = next.to_string();
                store::write_cursor(&cursor, &buf);
            }
        }
    }

    async fn send(
        &self,
        peer: &PeerKey,
        ctx: &ReplyCtx,
        msg: OutboundMessage,
    ) -> Result<Option<MsgHandle>, SendError> {
        let to = peer.chat.as_str();
        let ctx = self.context_for(peer, ctx);
        let ctx = ctx.as_deref();
        let text_item = |t: &str| json!({"type": ITEM_TEXT, "text_item": {"text": t}});
        match msg {
            OutboundMessage::Text { text, format } | OutboundMessage::Edit { text, format, .. } => {
                let t = match format {
                    TextFormat::Markdown => markdown::render(&text, MarkdownDialect::Plain),
                    TextFormat::Plain => text,
                };
                self.api.send_items(to, ctx, vec![text_item(&t)]).await?;
                Ok(None)
            }
            OutboundMessage::Choices { question, options } => {
                let mut t = question;
                for (i, o) in options.iter().enumerate() {
                    t.push_str(&format!("\n{}. {o}", i + 1));
                }
                if !options.is_empty() {
                    t.push_str("\n回复编号选择，或直接回答");
                }
                self.api.send_items(to, ctx, vec![text_item(&t)]).await?;
                Ok(None)
            }
            OutboundMessage::Typing => self.typing(to, ctx).await.map(|_| None),
            OutboundMessage::File { path, caption } => self
                .send_file(to, ctx, &path, caption.as_deref())
                .await
                .map(|_| None),
        }
    }

    async fn fetch_media(
        &self,
        media: &MediaRef,
        dir: &Path,
        max_bytes: u64,
    ) -> anyhow::Result<PathBuf> {
        if media.size.is_some_and(|s| s > max_bytes) {
            anyhow::bail!("file too large ({} bytes)", media.size.unwrap_or(0));
        }
        let m = media
            .data
            .get("media")
            .ok_or_else(|| anyhow::anyhow!("media item without CDN reference"))?;
        // image_item.aeskey (hex) wins over media.aes_key (base64).
        let key = media
            .data
            .get("aeskey")
            .and_then(Value::as_str)
            .or_else(|| m.get("aes_key").and_then(Value::as_str))
            .and_then(cdn::parse_key);
        let bytes = cdn::download(&self.api.http, m, key, max_bytes).await?;
        let name = media.name.clone().unwrap_or_else(|| "file".into());
        std::fs::create_dir_all(dir)?;
        let dest = dir.join(super::safe_file_name(&name));
        tokio::fs::write(&dest, &bytes).await?;
        Ok(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wx() -> Weixin {
        Weixin::new(
            reqwest::Client::new(),
            "",
            "tok",
            "bot1",
            api::Identity::new("bot", "2.4.10"),
        )
    }

    #[test]
    fn user_text_with_quote() {
        let m = json!({"message_id": 7_000_000_000_000_000_001u64, "from_user_id": "u1@im.wechat",
            "message_type": 1, "context_token": "ctx1",
            "item_list": [{"type": 1, "text_item": {"text": "再改一下"},
                           "ref_msg": {"message_item": {"type": 1, "text_item": {"text": "旧回答"}}}}]});
        let msg = wx().parse_message(&m).unwrap();
        assert_eq!(msg.text, "> 旧回答\n再改一下");
        assert_eq!(msg.msg_id, "7000000000000000001");
        assert_eq!(msg.peer.key(), "weixin:bot1:u1@im.wechat");
        assert_eq!(msg.reply_ctx, ReplyCtx(Some("ctx1".into())));
        assert!(msg.addressed);
    }

    #[test]
    fn bot_echoes_and_groups_are_not_addressed() {
        let echo = json!({"from_user_id": "bot", "message_type": 2,
            "item_list": [{"type": 1, "text_item": {"text": "hi"}}]});
        assert!(wx().parse_message(&echo).is_none());
        let group = json!({"from_user_id": "u2", "group_id": "g1", "message_type": 1,
            "item_list": [{"type": 1, "text_item": {"text": "hi"}}]});
        assert!(!wx().parse_message(&group).unwrap().addressed);
    }

    #[test]
    fn voice_uses_server_transcript_and_files_become_media() {
        let m = json!({"from_user_id": "u1", "message_type": 1, "item_list": [
            {"type": 3, "voice_item": {"text": "帮我看看日志"}},
            {"type": 4, "file_item": {"file_name": "a.log", "len": "12", "media": {"encrypt_query_param": "q"}}}]});
        let msg = wx().parse_message(&m).unwrap();
        assert_eq!(msg.text, "帮我看看日志");
        assert_eq!(msg.media.len(), 1);
        assert_eq!(msg.media[0].name.as_deref(), Some("a.log"));
        assert_eq!(msg.media[0].size, Some(12));
    }
}
