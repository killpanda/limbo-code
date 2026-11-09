//! The iLink bot HTTP API (`https://ilinkai.weixin.qq.com/ilink/bot/...`):
//! JSON over POST, `Authorization: Bearer <bot_token>`. Mirrors the
//! endpoints of Tencent's own reference plugin (`@tencent-weixin/openclaw-weixin`,
//! MIT): `getupdates` (long poll), `sendmessage`, `sendtyping`, `getconfig`,
//! `getuploadurl`, and the QR-code login pair `get_bot_qrcode` /
//! `get_qrcode_status`.

use std::time::Duration;

use base64::Engine;
use serde_json::{Value, json};

use crate::channels::SendError;

pub const DEFAULT_BASE_URL: &str = "https://ilinkai.weixin.qq.com";
pub const CDN_BASE_URL: &str = "https://novac2c.cdn.weixin.qq.com/c2c";
/// The server's "bot token expired" error code.
pub const STALE_TOKEN_ERRCODE: i64 = -14;
/// `MessageType`: messages from a user (bots' own messages are 2).
pub const MESSAGE_TYPE_USER: i64 = 1;
pub const MESSAGE_TYPE_BOT: i64 = 2;
pub const MESSAGE_STATE_FINISH: i64 = 2;
/// `MessageItemType`.
pub const ITEM_TEXT: i64 = 1;
pub const ITEM_IMAGE: i64 = 2;
pub const ITEM_VOICE: i64 = 3;
pub const ITEM_FILE: i64 = 4;
pub const ITEM_VIDEO: i64 = 5;
/// `UploadMediaType`.
pub const UPLOAD_IMAGE: i64 = 1;
pub const UPLOAD_FILE: i64 = 3;

pub const LONG_POLL_TIMEOUT: Duration = Duration::from_secs(35);
const API_TIMEOUT: Duration = Duration::from_secs(15);

/// Request identity sent with every call (`[channels.weixin]`).
#[derive(Debug, Clone)]
pub struct Identity {
    /// `iLink-App-Id` header.
    pub app_id: String,
    /// `iLink-App-ClientVersion` header (`major<<16 | minor<<8 | patch`).
    pub client_version: u32,
    /// `base_info.bot_agent`: who is calling, for Tencent's logs.
    pub bot_agent: String,
    pub channel_version: String,
}

impl Identity {
    pub fn new(app_id: &str, client_version: &str) -> Self {
        Self {
            app_id: app_id.to_string(),
            client_version: encode_version(client_version),
            bot_agent: format!("limbo/{}", env!("CARGO_PKG_VERSION")),
            channel_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn base_info(&self) -> Value {
        json!({"channel_version": self.channel_version, "bot_agent": self.bot_agent})
    }
}

/// `"2.4.10"` → `0x02040A`.
pub fn encode_version(v: &str) -> u32 {
    let mut parts = v
        .split('.')
        .map(|p| p.trim().parse::<u32>().unwrap_or(0) & 0xff);
    let a = parts.next().unwrap_or(0);
    let b = parts.next().unwrap_or(0);
    let c = parts.next().unwrap_or(0);
    (a << 16) | (b << 8) | c
}

/// `X-WECHAT-UIN`: a random uint32, decimal, base64-encoded.
fn random_uin() -> String {
    let n: u32 = rand::random();
    base64::engine::general_purpose::STANDARD.encode(n.to_string())
}

pub struct Api {
    pub http: reqwest::Client,
    pub base_url: String,
    pub token: String,
    pub id: Identity,
}

impl Api {
    fn endpoint(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url.trim_end_matches('/'))
    }

    fn request(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut rb = rb
            .header("AuthorizationType", "ilink_bot_token")
            .header("X-WECHAT-UIN", random_uin())
            .header("iLink-App-Id", &self.id.app_id)
            .header(
                "iLink-App-ClientVersion",
                self.id.client_version.to_string(),
            );
        if !self.token.is_empty() {
            rb = rb.bearer_auth(&self.token);
        }
        rb
    }

    /// POST `body` (+ `base_info`) and parse the JSON reply.
    pub async fn post(
        &self,
        path: &str,
        mut body: Value,
        timeout: Duration,
    ) -> Result<Value, SendError> {
        if let Some(o) = body.as_object_mut() {
            o.insert("base_info".into(), self.id.base_info());
        }
        let resp = self
            .request(self.http.post(self.endpoint(path)))
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    SendError::Transient("timeout".into())
                } else {
                    SendError::Transient(e.to_string())
                }
            })?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| SendError::Transient(e.to_string()))?;
        if status.is_server_error() {
            return Err(SendError::Transient(format!("HTTP {status}")));
        }
        if !status.is_success() {
            return Err(SendError::Other(format!(
                "HTTP {status}: {}",
                text.chars().take(200).collect::<String>()
            )));
        }
        let v: Value = serde_json::from_str(&text)
            .map_err(|_| SendError::Transient(format!("bad JSON from {path}")))?;
        check_ret(&v)?;
        Ok(v)
    }

    pub async fn get_updates(&self, buf: &str) -> Result<Value, SendError> {
        match self
            .post(
                "ilink/bot/getupdates",
                json!({"get_updates_buf": buf}),
                LONG_POLL_TIMEOUT + Duration::from_secs(5),
            )
            .await
        {
            // A client-side timeout of the long poll is an empty answer.
            Err(SendError::Transient(t)) if t == "timeout" => {
                Ok(json!({"ret": 0, "msgs": [], "get_updates_buf": buf}))
            }
            r => r,
        }
    }

    pub async fn send_items(
        &self,
        to: &str,
        context_token: Option<&str>,
        items: Vec<Value>,
    ) -> Result<Value, SendError> {
        let client_id = format!("limbo-{}", crate::agent::new_session_id());
        let mut msg = json!({
            "from_user_id": "",
            "to_user_id": to,
            "client_id": client_id,
            "message_type": MESSAGE_TYPE_BOT,
            "message_state": MESSAGE_STATE_FINISH,
            "item_list": items,
        });
        if let Some(t) = context_token {
            msg["context_token"] = json!(t);
        }
        self.post("ilink/bot/sendmessage", json!({"msg": msg}), API_TIMEOUT)
            .await
    }

    pub async fn get_config(
        &self,
        user: &str,
        context_token: Option<&str>,
    ) -> Result<Value, SendError> {
        self.post(
            "ilink/bot/getconfig",
            json!({"ilink_user_id": user, "context_token": context_token}),
            Duration::from_secs(10),
        )
        .await
    }

    pub async fn send_typing(&self, user: &str, ticket: &str) -> Result<Value, SendError> {
        self.post(
            "ilink/bot/sendtyping",
            json!({"ilink_user_id": user, "typing_ticket": ticket, "status": 1}),
            Duration::from_secs(10),
        )
        .await
    }

    pub async fn get_upload_url(&self, req: Value) -> Result<Value, SendError> {
        self.post("ilink/bot/getuploadurl", req, API_TIMEOUT).await
    }

    /// Unauthenticated GET (QR login).
    pub async fn get(&self, path_and_query: &str, timeout: Duration) -> Result<Value, SendError> {
        let resp = self
            .http
            .get(self.endpoint(path_and_query))
            .header("iLink-App-Id", &self.id.app_id)
            .header(
                "iLink-App-ClientVersion",
                self.id.client_version.to_string(),
            )
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    SendError::Transient("timeout".into())
                } else {
                    SendError::Transient(e.to_string())
                }
            })?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(SendError::Other(format!("HTTP {status}")));
        }
        serde_json::from_str(&text).map_err(|_| SendError::Other("bad JSON".into()))
    }
}

/// `ret` / `errcode` → an error.
pub fn check_ret(v: &Value) -> Result<(), SendError> {
    let ret = v.get("ret").and_then(Value::as_i64).unwrap_or(0);
    let errcode = v.get("errcode").and_then(Value::as_i64).unwrap_or(0);
    if ret == 0 && errcode == 0 {
        return Ok(());
    }
    let msg = v
        .get("errmsg")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if ret == STALE_TOKEN_ERRCODE || errcode == STALE_TOKEN_ERRCODE {
        return Err(SendError::AuthExpired(format!("bot token expired ({msg})")));
    }
    let lower = msg.to_lowercase();
    if lower.contains("context") || (lower.contains("token") && lower.contains("expire")) {
        return Err(SendError::ContextExpired);
    }
    if lower.contains("freq") || lower.contains("limit") || msg.contains("频") {
        return Err(SendError::RateLimited(Duration::from_secs(5)));
    }
    Err(SendError::Other(format!(
        "ret={ret} errcode={errcode} {msg}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_encode_like_the_reference_plugin() {
        assert_eq!(encode_version("2.4.10"), 0x02040A);
        assert_eq!(encode_version("1.0"), 0x010000);
    }

    #[test]
    fn ret_codes() {
        assert!(check_ret(&json!({"ret": 0})).is_ok());
        assert!(matches!(
            check_ret(&json!({"ret": -14, "errmsg": "session timeout"})),
            Err(SendError::AuthExpired(_))
        ));
        assert!(matches!(
            check_ret(&json!({"ret": 1, "errmsg": "context token invalid"})),
            Err(SendError::ContextExpired)
        ));
        assert!(matches!(
            check_ret(&json!({"ret": 3})),
            Err(SendError::Other(_))
        ));
    }
}
