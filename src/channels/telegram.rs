//! Telegram through the official Bot API (`https://core.telegram.org/bots/api`),
//! hand-written on reqwest: `getUpdates` long polling, `sendMessage` /
//! `editMessageText` (HTML parse mode), `sendChatAction`, `sendDocument`,
//! `getFile` and inline-keyboard callbacks for `ask_user` options.
//!
//! Rate limits are reported by Telegram itself (`429` + `retry_after`) and
//! become [`SendError::RateLimited`]; the gateway's outbox waits and retries.
//!
//! 🦀 Rust notes
//! - Responses are parsed as `serde_json::Value` and read with `.get()` /
//!   `.as_*()`: only a dozen fields are used, so typed structs for the whole
//!   Bot API would be mostly dead weight.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{
    Channel, ChannelCaps, ChannelKind, InboundMessage, MarkdownDialect, MediaKind, MediaRef,
    MsgHandle, OutboundMessage, PeerKey, ReplyCtx, SendError, TextFormat, markdown, store,
};

pub const DEFAULT_API_BASE: &str = "https://api.telegram.org";
const POLL_TIMEOUT_SECS: u64 = 30;

pub struct Telegram {
    http: reqwest::Client,
    api_base: String,
    token: String,
    bot_id: String,
    bot_username: String,
    /// Accept group messages that mention the bot (`[channels.telegram] groups`).
    groups: bool,
}

/// The parts of a Bot API reply the adapter needs.
fn parse_reply(status: reqwest::StatusCode, body: &str) -> Result<Value, SendError> {
    let v: Value = serde_json::from_str(body)
        .map_err(|_| SendError::Transient(format!("HTTP {status}: {}", truncate(body, 200))))?;
    if v.get("ok").and_then(Value::as_bool) == Some(true) {
        return Ok(v.get("result").cloned().unwrap_or(Value::Null));
    }
    let code = v.get("error_code").and_then(Value::as_i64).unwrap_or(0);
    let desc = v
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("unknown error")
        .to_string();
    Err(match code {
        429 => SendError::RateLimited(Duration::from_secs(
            v.pointer("/parameters/retry_after")
                .and_then(Value::as_u64)
                .unwrap_or(3),
        )),
        401 | 404 if desc.contains("Unauthorized") || desc.contains("Not Found") => {
            SendError::AuthExpired(desc)
        }
        c if c >= 500 => SendError::Transient(desc),
        _ => SendError::Other(desc),
    })
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Build the reqwest client (optional proxy: `api.telegram.org` is often
/// blocked and needs one).
pub fn http_client(proxy: Option<&str>, timeout: Duration) -> anyhow::Result<reqwest::Client> {
    let mut b = reqwest::Client::builder()
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(15));
    if let Some(p) = proxy.filter(|p| !p.trim().is_empty()) {
        b = b.proxy(reqwest::Proxy::all(p.trim())?);
    }
    Ok(b.build()?)
}

/// `getMe`: (bot id, username). Used by login and at gateway start.
pub async fn get_me(
    http: &reqwest::Client,
    api_base: &str,
    token: &str,
) -> Result<(String, String), SendError> {
    let url = format!("{}/bot{token}/getMe", api_base.trim_end_matches('/'));
    let resp = http
        .get(&url)
        .send()
        .await
        .map_err(|e| SendError::Transient(e.without_url().to_string()))?;
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    let me = parse_reply(status, &body)?;
    Ok((
        me.get("id").map(id_string).unwrap_or_default(),
        me.get("username")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    ))
}

fn id_string(v: &Value) -> String {
    match v {
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

impl Telegram {
    pub fn new(
        http: reqwest::Client,
        api_base: &str,
        token: &str,
        bot_id: &str,
        bot_username: &str,
        groups: bool,
    ) -> Self {
        Self {
            http,
            api_base: api_base.trim_end_matches('/').to_string(),
            token: token.to_string(),
            bot_id: bot_id.to_string(),
            bot_username: bot_username.to_string(),
            groups,
        }
    }

    fn url(&self, method: &str) -> String {
        format!("{}/bot{}/{method}", self.api_base, self.token)
    }

    async fn call(&self, method: &str, body: Value, timeout: Duration) -> Result<Value, SendError> {
        let resp = self
            .http
            .post(self.url(method))
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            // without_url: the URL carries the bot token.
            .map_err(|e| SendError::Transient(e.without_url().to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| SendError::Transient(e.without_url().to_string()))?;
        parse_reply(status, &text)
    }

    fn cursor_name(&self) -> String {
        format!("telegram-{}", self.bot_id)
    }

    /// Turn one `Update` into an inbound message (None: not for us).
    pub fn parse_update(&self, u: &Value) -> Option<InboundMessage> {
        if let Some(cb) = u.get("callback_query") {
            return self.parse_callback(cb);
        }
        let m = u.get("message").or_else(|| u.get("edited_message"))?;
        let chat = m.get("chat")?;
        let chat_id = id_string(chat.get("id")?);
        let chat_type = chat.get("type").and_then(Value::as_str).unwrap_or("");
        let is_group = chat_type != "private";
        let from = m.get("from")?;
        if from.get("is_bot").and_then(Value::as_bool) == Some(true) {
            return None;
        }
        let sender = id_string(from.get("id")?);
        let sender_name = from
            .get("username")
            .or_else(|| from.get("first_name"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let thread = (m.get("is_topic_message").and_then(Value::as_bool) == Some(true))
            .then(|| m.get("message_thread_id").map(id_string))
            .flatten();
        let mut text = m
            .get("text")
            .or_else(|| m.get("caption"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mention = format!("@{}", self.bot_username);
        let mentioned = !self.bot_username.is_empty() && text.contains(&mention);
        if mentioned {
            text = text.replace(&mention, "").trim().to_string();
        }
        let replied_to_bot = m
            .pointer("/reply_to_message/from/id")
            .map(id_string)
            .is_some_and(|id| id == self.bot_id);
        // `/cmd@thisbot` in groups.
        if let Some(rest) = text.strip_prefix('/')
            && let Some((cmd, tail)) = rest.split_once(' ').or(Some((rest, "")))
            && let Some((c, bot)) = cmd.split_once('@')
            && bot.eq_ignore_ascii_case(&self.bot_username)
        {
            text = format!("/{c} {tail}").trim().to_string();
        }
        let mut media = Vec::new();
        if let Some(photos) = m.get("photo").and_then(Value::as_array)
            && let Some(best) = photos.last()
        {
            media.push(MediaRef {
                kind: MediaKind::Image,
                name: None,
                size: best.get("file_size").and_then(Value::as_u64),
                data: json!({"file_id": best.get("file_id")}),
            });
        }
        for (key, kind) in [
            ("document", MediaKind::File),
            ("voice", MediaKind::Voice),
            ("audio", MediaKind::Voice),
            ("video", MediaKind::Video),
        ] {
            if let Some(d) = m.get(key) {
                media.push(MediaRef {
                    kind,
                    name: d.get("file_name").and_then(Value::as_str).map(String::from),
                    size: d.get("file_size").and_then(Value::as_u64),
                    data: json!({"file_id": d.get("file_id")}),
                });
            }
        }
        if text.is_empty() && media.is_empty() {
            return None;
        }
        Some(InboundMessage {
            peer: PeerKey {
                channel: ChannelKind::Telegram,
                account: self.bot_id.clone(),
                chat: chat_id,
                thread,
            },
            sender,
            sender_name,
            msg_id: format!(
                "{}:{}",
                m.pointer("/chat/id").map(id_string).unwrap_or_default(),
                m.get("message_id").map(id_string).unwrap_or_default()
            ),
            text,
            media,
            reply_ctx: ReplyCtx::default(),
            is_group,
            addressed: !is_group || (self.groups && (mentioned || replied_to_bot)),
        })
    }

    /// An inline-keyboard press: becomes the option number as text, the
    /// same answer `ask_user` accepts when typed.
    fn parse_callback(&self, cb: &Value) -> Option<InboundMessage> {
        let data = cb.get("data").and_then(Value::as_str)?.to_string();
        let m = cb.get("message")?;
        let chat = m.get("chat")?;
        let is_group = chat.get("type").and_then(Value::as_str) != Some("private");
        let from = cb.get("from")?;
        let thread = (m.get("is_topic_message").and_then(Value::as_bool) == Some(true))
            .then(|| m.get("message_thread_id").map(id_string))
            .flatten();
        Some(InboundMessage {
            peer: PeerKey {
                channel: ChannelKind::Telegram,
                account: self.bot_id.clone(),
                chat: id_string(chat.get("id")?),
                thread,
            },
            sender: id_string(from.get("id")?),
            sender_name: from
                .get("username")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            msg_id: format!("cb:{}", cb.get("id").map(id_string).unwrap_or_default()),
            text: data,
            media: vec![],
            reply_ctx: ReplyCtx(cb.get("id").map(id_string)),
            is_group,
            addressed: true,
        })
    }

    fn base_body(peer: &PeerKey) -> serde_json::Map<String, Value> {
        let mut b = serde_json::Map::new();
        b.insert("chat_id".into(), json!(peer.chat));
        if let Some(t) = &peer.thread {
            b.insert(
                "message_thread_id".into(),
                json!(t.parse::<i64>().unwrap_or(0)),
            );
        }
        b
    }

    /// Send or edit text; on an HTML parse error, retry once as plain text.
    async fn send_text(
        &self,
        peer: &PeerKey,
        edit: Option<&MsgHandle>,
        text: &str,
        format: TextFormat,
    ) -> Result<Option<MsgHandle>, SendError> {
        let mut body = Self::base_body(peer);
        let method = match edit {
            Some(h) => {
                body.remove("message_thread_id");
                body.insert("message_id".into(), json!(h.0.parse::<i64>().unwrap_or(0)));
                "editMessageText"
            }
            None => "sendMessage",
        };
        body.insert("link_preview_options".into(), json!({"is_disabled": true}));
        let html = format == TextFormat::Markdown;
        let rendered = if html {
            markdown::render(text, MarkdownDialect::TelegramHtml)
        } else {
            text.to_string()
        };
        body.insert("text".into(), json!(rendered));
        if html {
            body.insert("parse_mode".into(), json!("HTML"));
        }
        let r = match self
            .call(method, Value::Object(body.clone()), Duration::from_secs(20))
            .await
        {
            Err(SendError::Other(d)) if html && d.contains("can't parse entities") => {
                body.remove("parse_mode");
                body.insert(
                    "text".into(),
                    json!(markdown::render(text, MarkdownDialect::Plain)),
                );
                self.call(method, Value::Object(body), Duration::from_secs(20))
                    .await
            }
            // Editing to identical text is not an error for us.
            Err(SendError::Other(d)) if d.contains("message is not modified") => {
                return Ok(edit.cloned());
            }
            r => r,
        }?;
        Ok(r.get("message_id")
            .map(id_string)
            .map(MsgHandle)
            .or_else(|| edit.cloned()))
    }

    async fn send_file(
        &self,
        peer: &PeerKey,
        path: &Path,
        caption: Option<&str>,
    ) -> Result<Option<MsgHandle>, SendError> {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| SendError::Other(format!("cannot read {}: {e}", path.display())))?;
        if bytes.len() > 50 * 1024 * 1024 {
            return Err(SendError::Other(
                "Telegram bots can send files up to 50 MB".into(),
            ));
        }
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file".into());
        let mut fields = vec![("chat_id", peer.chat.clone())];
        if let Some(t) = &peer.thread {
            fields.push(("message_thread_id", t.clone()));
        }
        if let Some(c) = caption.filter(|c| !c.is_empty()) {
            fields.push(("caption", truncate(c, 1000)));
        }
        let (ct, body) = super::multipart_body(&fields, "document", &name, &bytes);
        let resp = self
            .http
            .post(self.url("sendDocument"))
            .timeout(Duration::from_secs(300))
            .header(reqwest::header::CONTENT_TYPE, ct)
            .body(body)
            .send()
            .await
            .map_err(|e| SendError::Transient(e.without_url().to_string()))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let r = parse_reply(status, &text)?;
        Ok(r.get("message_id").map(id_string).map(MsgHandle))
    }
}

#[async_trait]
impl Channel for Telegram {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Telegram
    }
    fn account(&self) -> String {
        self.bot_id.clone()
    }
    fn caps(&self) -> ChannelCaps {
        ChannelCaps {
            max_text_chars: 4096,
            edit: true,
            markdown: MarkdownDialect::TelegramHtml,
            min_send_interval: Duration::from_millis(1100),
            typing_ttl: Duration::from_secs(5),
        }
    }

    async fn run(
        &self,
        inbox: mpsc::Sender<InboundMessage>,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        let cursor = self.cursor_name();
        let mut offset: i64 = store::read_cursor(&cursor)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let mut backoff = 1u64;
        loop {
            let body = json!({
                "offset": offset,
                "timeout": POLL_TIMEOUT_SECS,
                "allowed_updates": ["message", "callback_query"],
            });
            let poll = self.call(
                "getUpdates",
                body,
                Duration::from_secs(POLL_TIMEOUT_SECS + 15),
            );
            let res = tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                r = poll => r,
            };
            let updates = match res {
                Ok(v) => {
                    backoff = 1;
                    v.as_array().cloned().unwrap_or_default()
                }
                Err(SendError::AuthExpired(d)) => {
                    anyhow::bail!(
                        "Telegram rejected the bot token ({d}); run `limbo channels login telegram`"
                    )
                }
                Err(SendError::RateLimited(d)) => {
                    tokio::time::sleep(d).await;
                    continue;
                }
                Err(e) => {
                    eprintln!("telegram: getUpdates failed: {e}; retrying in {backoff}s");
                    tokio::select! {
                        _ = cancel.cancelled() => return Ok(()),
                        _ = tokio::time::sleep(Duration::from_secs(backoff)) => {}
                    }
                    backoff = (backoff * 2).min(60);
                    continue;
                }
            };
            for u in updates {
                let id = u.get("update_id").and_then(Value::as_i64).unwrap_or(0);
                if let Some(cb) = u.get("callback_query")
                    && let Some(cid) = cb.get("id")
                {
                    let _ = self
                        .call(
                            "answerCallbackQuery",
                            json!({"callback_query_id": cid}),
                            Duration::from_secs(10),
                        )
                        .await;
                }
                if let Some(msg) = self.parse_update(&u)
                    && inbox.send(msg).await.is_err()
                {
                    return Ok(());
                }
                // The update is in the gateway's hands: never fetch it again.
                offset = offset.max(id + 1);
                store::write_cursor(&cursor, &offset.to_string());
            }
        }
    }

    async fn send(
        &self,
        peer: &PeerKey,
        _ctx: &ReplyCtx,
        msg: OutboundMessage,
    ) -> Result<Option<MsgHandle>, SendError> {
        match msg {
            OutboundMessage::Text { text, format } => {
                self.send_text(peer, None, &text, format).await
            }
            OutboundMessage::Edit {
                handle,
                text,
                format,
            } => self.send_text(peer, Some(&handle), &text, format).await,
            OutboundMessage::File { path, caption } => {
                self.send_file(peer, &path, caption.as_deref()).await
            }
            OutboundMessage::Typing => {
                let mut body = Self::base_body(peer);
                body.insert("action".into(), json!("typing"));
                self.call(
                    "sendChatAction",
                    Value::Object(body),
                    Duration::from_secs(10),
                )
                .await
                .map(|_| None)
            }
            OutboundMessage::Choices { question, options } => {
                let mut body = Self::base_body(peer);
                body.insert("text".into(), json!(question));
                let rows: Vec<Value> = options
                    .iter()
                    .enumerate()
                    .map(|(i, o)| {
                        json!([{"text": truncate(&format!("{}. {o}", i + 1), 60),
                                "callback_data": (i + 1).to_string()}])
                    })
                    .collect();
                body.insert("reply_markup".into(), json!({"inline_keyboard": rows}));
                let r = self
                    .call("sendMessage", Value::Object(body), Duration::from_secs(20))
                    .await?;
                Ok(r.get("message_id").map(id_string).map(MsgHandle))
            }
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
        let file_id = media
            .data
            .get("file_id")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("missing file_id"))?;
        let f = self
            .call(
                "getFile",
                json!({"file_id": file_id}),
                Duration::from_secs(20),
            )
            .await?;
        let fpath = f
            .get("file_path")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("getFile returned no file_path"))?;
        let url = format!("{}/file/bot{}/{fpath}", self.api_base, self.token);
        let resp = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(300))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("download failed: {}", e.without_url()))?;
        if !resp.status().is_success() {
            anyhow::bail!("download failed: HTTP {}", resp.status());
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("download failed: {}", e.without_url()))?;
        if bytes.len() as u64 > max_bytes {
            anyhow::bail!("file too large ({} bytes)", bytes.len());
        }
        let name = media.name.clone().unwrap_or_else(|| {
            Path::new(fpath)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "file".into())
        });
        std::fs::create_dir_all(dir)?;
        let dest = dir.join(super::safe_file_name(&name));
        tokio::fs::write(&dest, &bytes).await?;
        Ok(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bot() -> Telegram {
        Telegram::new(
            reqwest::Client::new(),
            DEFAULT_API_BASE,
            "T",
            "999",
            "limbo_bot",
            true,
        )
    }

    #[test]
    fn private_text_message() {
        let u = json!({"update_id": 5, "message": {
            "message_id": 11, "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": false, "username": "jack"}, "text": "hello"}});
        let m = bot().parse_update(&u).unwrap();
        assert_eq!(m.peer.key(), "telegram:999:42");
        assert_eq!(m.sender, "42");
        assert_eq!(m.msg_id, "42:11");
        assert!(m.addressed && !m.is_group);
    }

    #[test]
    fn group_needs_mention_and_strips_it() {
        let plain = json!({"update_id": 6, "message": {
            "message_id": 12, "chat": {"id": -100, "type": "supergroup"},
            "from": {"id": 42, "is_bot": false}, "text": "just chatting"}});
        assert!(!bot().parse_update(&plain).unwrap().addressed);
        let mention = json!({"update_id": 7, "message": {
            "message_id": 13, "chat": {"id": -100, "type": "supergroup"},
            "from": {"id": 42, "is_bot": false}, "text": "@limbo_bot fix the build"}});
        let m = bot().parse_update(&mention).unwrap();
        assert!(m.addressed);
        assert_eq!(m.text, "fix the build");
        let cmd = json!({"update_id": 8, "message": {
            "message_id": 14, "chat": {"id": -100, "type": "supergroup"},
            "from": {"id": 42, "is_bot": false}, "text": "/new@limbo_bot"}});
        assert_eq!(bot().parse_update(&cmd).unwrap().text, "/new");
    }

    #[test]
    fn photo_and_callback() {
        let u = json!({"update_id": 9, "message": {
            "message_id": 15, "chat": {"id": 42, "type": "private"},
            "from": {"id": 42, "is_bot": false}, "caption": "看这个",
            "photo": [{"file_id": "small", "file_size": 10}, {"file_id": "big", "file_size": 99}]}});
        let m = bot().parse_update(&u).unwrap();
        assert_eq!(m.text, "看这个");
        assert_eq!(m.media[0].data["file_id"], "big");
        let cb = json!({"update_id": 10, "callback_query": {"id": "c1", "data": "2",
            "from": {"id": 42}, "message": {"message_id": 3, "chat": {"id": 42, "type": "private"}}}});
        let m = bot().parse_update(&cb).unwrap();
        assert_eq!(m.text, "2");
    }

    #[test]
    fn errors_map_to_send_errors() {
        let r = parse_reply(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"ok":false,"error_code":429,"description":"Too Many Requests","parameters":{"retry_after":7}}"#,
        );
        assert!(matches!(r, Err(SendError::RateLimited(d)) if d == Duration::from_secs(7)));
        let r = parse_reply(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"ok":false,"error_code":401,"description":"Unauthorized"}"#,
        );
        assert!(matches!(r, Err(SendError::AuthExpired(_))));
    }
}
