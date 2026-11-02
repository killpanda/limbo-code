//! LLM clients: one HTTP client, three wire dialects.
//!
//! The flow for every request is the same:
//! 1. the dialect module turns our `Message`s into a JSON body,
//! 2. we POST it and check the HTTP status (retrying 429/5xx/network
//!    failures with exponential backoff *before* anything was streamed),
//! 3. the SSE stream is fed into the dialect's [`Decoder`], which turns
//!    provider events into our provider-neutral [`LlmEvent`]s.
//!
//! 🦀 Rust notes
//! - [`Decoder`] is a *trait*: an interface each dialect implements. We hold
//!   one as `Box<dyn Decoder>` so the streaming code does not care which
//!   dialect it is talking to (dynamic dispatch).
//! - `async_stream`-style generators are not in stable Rust yet, so we build
//!   the output stream with a channel: a spawned task pushes events into
//!   `tx`, the caller reads from `rx`.

pub mod anthropic;
pub mod catalog;
pub mod openai;
pub mod responses;
pub mod sse;
pub mod usage;

use std::collections::HashMap;
use std::time::{Duration, Instant};

use base64::Engine;
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::config::Config;
use crate::models::{Attachment, LlmEvent, Message};
use catalog::{Dialect, ModelSpec};
use sse::SseEvent;

/// Errors from an LLM request, classified so the agent can decide whether
/// to retry, compact, or give up.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP {status}: {message}")]
    Http {
        status: u16,
        message: String,
        retry_after: Option<f64>,
    },
    #[error("network error: {0}")]
    Transport(String),
    #[error("provider overloaded: {0}")]
    Overloaded(String),
    #[error("{0}")]
    Other(String),
}

const OVERFLOW_PATTERNS: &[&str] = &[
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "prompt is too long",
    "input is too long",
    "too many tokens",
    "reduce the length of the messages",
    "exceeds the maximum number of tokens",
    "model_context_window_exceeded",
];

impl LlmError {
    /// Whether retrying the same request could succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::Http { status, .. } => matches!(status, 408 | 429) || *status >= 500,
            LlmError::Transport(_) | LlmError::Overloaded(_) => true,
            LlmError::Other(_) => false,
        }
    }

    /// Whether the prompt is too big for the model (recoverable by compacting).
    pub fn is_context_overflow(&self) -> bool {
        if let LlmError::Http { status, .. } = self {
            if *status == 413 {
                return true;
            }
            if *status != 400 && *status != 422 {
                return false;
            }
        }
        let text = self.to_string().to_lowercase();
        if text.contains("per minute") || text.contains("rate limit") {
            return false;
        }
        OVERFLOW_PATTERNS.iter().any(|p| text.contains(p))
    }

    /// A short, user-facing explanation.
    pub fn friendly(&self) -> String {
        match self {
            LlmError::Http {
                status: 401 | 403, ..
            } => format!("认证失败（API key 无效或无权限）：{self}"),
            LlmError::Http { status: 408, .. } => {
                "请求超时，已自动重试仍失败，请检查网络后重发上一条消息".into()
            }
            LlmError::Http { status: 429, .. } => {
                "服务限流，已自动重试仍失败，可稍后重发上一条消息".into()
            }
            LlmError::Http { status, .. } if *status >= 500 => {
                format!("模型服务异常（HTTP {status}），已自动重试仍失败，可稍后重发上一条消息")
            }
            other => other.to_string(),
        }
    }
}

/// Turns one dialect's SSE events into [`LlmEvent`]s.
pub trait Decoder: Send {
    /// Handle one SSE event. Returning `Ok(true)` means "the stream is done".
    fn on_event(&mut self, ev: &SseEvent, out: &mut Vec<LlmEvent>) -> Result<bool, LlmError>;
    /// Flush anything still open (tool calls cut off by a dropped stream)
    /// and emit the final `Done` metadata.
    fn finish(&mut self, out: &mut Vec<LlmEvent>);
}

/// A built request: path relative to the base URL, plus JSON body.
pub struct Request {
    pub path: &'static str,
    pub body: Value,
}

/// Called with every request body before it is sent (tracing).
pub type RequestHook = std::sync::Arc<dyn Fn(&Value) + Send + Sync>;
/// The client the agent talks to. Cheap to clone (reqwest's client is an
/// `Arc` inside).
#[derive(Clone)]
pub struct LlmClient {
    pub spec: ModelSpec,
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    headers: HashMap<String, String>,
    pub temperature: f64,
    pub max_tokens: u32,
    pub thinking_effort: Option<String>,
    max_retries: u32,
    retry_base_delay: f64,
    /// Responses API prompt cache bucket (the session id).
    pub prompt_cache_key: Option<String>,
    /// Called with every request body before it is sent (tracing).
    pub on_request: Option<RequestHook>,
}

impl LlmClient {
    pub fn new(config: &Config) -> anyhow::Result<Self> {
        Self::for_model(config, &config.llm.model)
    }

    pub fn for_model(config: &Config, model: &str) -> anyhow::Result<Self> {
        let spec = catalog::resolve_model(model);
        let api_key = catalog::resolve_api_key(&spec, config)
            .ok_or_else(|| anyhow::anyhow!(catalog::missing_api_key_message(&spec, config)))?;
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs_f64(config.llm.connect_timeout))
            .read_timeout(Duration::from_secs_f64(config.llm.timeout))
            .build()?;
        Ok(Self {
            base_url: catalog::resolve_base_url(&spec, config),
            headers: catalog::resolve_headers(&spec, config),
            max_tokens: config.llm.max_tokens.unwrap_or(spec.max_tokens),
            spec,
            http,
            api_key,
            temperature: config.llm.temperature,
            thinking_effort: config.llm.thinking_effort.clone(),
            max_retries: config.llm.max_retries,
            retry_base_delay: config.llm.retry_base_delay,
            prompt_cache_key: None,
            on_request: None,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.spec.id
    }

    fn build(&self, messages: &[Message], tools: &[Value]) -> Request {
        match self.spec.provider.dialect {
            Dialect::OpenAiCompletions => openai::build_request(self, messages, tools),
            Dialect::AnthropicMessages => anthropic::build_request(self, messages, tools),
            Dialect::OpenAiResponses => responses::build_request(self, messages, tools),
        }
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        match self.spec.provider.dialect {
            Dialect::OpenAiCompletions => Box::new(openai::OpenAiDecoder::default()),
            Dialect::AnthropicMessages => Box::new(anthropic::AnthropicDecoder::default()),
            Dialect::OpenAiResponses => Box::new(responses::ResponsesDecoder::default()),
        }
    }

    fn url(&self, path: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        // The Anthropic path starts with /v1; avoid doubling it when the
        // configured base already ends in /v1.
        if path.starts_with("/v1/") && base.ends_with("/v1") {
            format!("{}{}", base.trim_end_matches("/v1"), path)
        } else {
            format!("{base}{path}")
        }
    }

    /// Send one POST, retrying retryable failures with backoff.
    async fn send(&self, req: &Request) -> Result<reqwest::Response, LlmError> {
        let mut attempt = 0;
        loop {
            let mut builder = self.http.post(self.url(req.path)).json(&req.body);
            builder = match self.spec.provider.dialect {
                Dialect::AnthropicMessages => builder
                    .header("x-api-key", &self.api_key)
                    .header("anthropic-version", "2023-06-01"),
                _ => builder.bearer_auth(&self.api_key),
            };
            for (k, v) in &self.headers {
                builder = builder.header(k, v);
            }
            let result: Result<(), LlmError> = match builder.send().await {
                Ok(resp) if resp.status().is_success() => return Ok(resp),
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.parse::<f64>().ok());
                    let body = resp.text().await.unwrap_or_default();
                    Err(LlmError::Http {
                        status,
                        message: truncate_chars(&body, 2000),
                        retry_after,
                    })
                }
                Err(e) => Err(LlmError::Transport(e.to_string())),
            };
            let err = result.unwrap_err();
            if !err.is_retryable() || attempt >= self.max_retries {
                return Err(err);
            }
            let hinted = if let LlmError::Http {
                retry_after: Some(s),
                ..
            } = &err
            {
                Some(*s)
            } else {
                None
            };
            let backoff = hinted
                .unwrap_or(self.retry_base_delay * 2f64.powi(attempt as i32))
                .min(30.0);
            tokio::time::sleep(Duration::from_secs_f64(backoff)).await;
            attempt += 1;
        }
    }

    /// Stream one completion. Events arrive on the returned channel; an
    /// `Err` item ends the stream.
    pub fn chat(
        &self,
        messages: Vec<Message>,
        tools: Vec<Value>,
    ) -> mpsc::Receiver<Result<LlmEvent, LlmError>> {
        let (tx, rx) = mpsc::channel(256);
        let client = self.clone();
        tokio::spawn(async move {
            let start = Instant::now();
            let req = client.build(&messages, &tools);
            if let Some(hook) = &client.on_request {
                hook(&req.body);
            }
            let resp = match client.send(&req).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };
            let mut decoder = client.decoder();
            let mut stream = Box::pin(sse::sse_stream(Box::pin(resp.bytes_stream())));
            let mut out = Vec::new();
            let mut ttft = None;
            while let Some(item) = stream.next().await {
                let ev = match item {
                    Ok(ev) => ev,
                    Err(e) => {
                        let _ = tx.send(Err(LlmError::Transport(e.to_string()))).await;
                        return;
                    }
                };
                ttft.get_or_insert_with(|| start.elapsed().as_secs_f64());
                let done = match decoder.on_event(&ev, &mut out) {
                    Ok(d) => d,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };
                for e in out.drain(..) {
                    if tx.send(Ok(e)).await.is_err() {
                        return; // receiver dropped: the turn was cancelled
                    }
                }
                if done {
                    break;
                }
            }
            decoder.finish(&mut out);
            for mut e in out.drain(..) {
                if let LlmEvent::Done(meta) = &mut e {
                    meta.ttft = ttft;
                    meta.duration = Some(start.elapsed().as_secs_f64());
                }
                let _ = tx.send(Ok(e)).await;
            }
        });
        rx
    }
}

/// `(base64, mime)` for an image attachment; `None` when the file is gone.
pub fn encode_image(att: &Attachment) -> Option<(String, String)> {
    if att.kind != "image" {
        return None;
    }
    let bytes = std::fs::read(&att.path).ok()?;
    let mime = att.mime.clone().unwrap_or_else(|| "image/png".into());
    Some((
        base64::engine::general_purpose::STANDARD.encode(bytes),
        mime,
    ))
}

/// Parse streamed tool-call JSON; malformed JSON is kept for the model to see.
pub fn parse_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Object(Default::default());
    }
    serde_json::from_str(raw)
        .unwrap_or_else(|e| serde_json::json!({"raw_arguments": raw, "parse_error": e.to_string()}))
}

/// Serialize tool-call arguments as the JSON *string* OpenAI-style APIs want.
pub fn arguments_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_classification() {
        let e = LlmError::Http {
            status: 400,
            message: "maximum context length is 128k".into(),
            retry_after: None,
        };
        assert!(e.is_context_overflow());
        let rl = LlmError::Http {
            status: 429,
            message: "too many tokens per minute".into(),
            retry_after: None,
        };
        assert!(!rl.is_context_overflow());
        assert!(rl.is_retryable());
        assert!(
            LlmError::Http {
                status: 413,
                message: String::new(),
                retry_after: None
            }
            .is_context_overflow()
        );
    }

    #[test]
    fn malformed_arguments_are_kept() {
        let v = parse_arguments("{\"a\":");
        assert_eq!(v["raw_arguments"], "{\"a\":");
    }
}
