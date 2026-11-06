//! `web_search`: ranked web results from the configured provider
//! (`[web_search] provider`; DeepSeek's search API today).
//!
//! The tool is only registered when the provider has a key — a tool the
//! model cannot use successfully is noise in every prompt.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::truncate::{MAX_OUTPUT_BYTES, truncate_output};
use super::{Tool, ToolContext, arg_i64, arg_str};
use crate::config::Config;
use crate::llm::catalog::{resolve_api_key, resolve_model};
use crate::models::ToolResult;

pub const MAX_QUERY_CHARS: usize = 128;
const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub summary: String,
    pub site_name: Option<String>,
    pub date: Option<String>,
}

/// One search backend (only DeepSeek so far — add a variant for more).
#[derive(Debug, Clone)]
pub struct DeepSeekSearch {
    pub api_key: String,
    pub url: String,
    pub timeout: f64,
}

impl DeepSeekSearch {
    pub const MAX_COUNT: u32 = 50;
    pub const QUERY_HELP: &str = " Query operators: site:example.com (combine with OR), -site:example.com, -term, \
\"exact phrase\", lang:zh,en, after:2025-01-01, before:2026.";

    pub async fn search(&self, query: &str, count: u32) -> Result<Vec<SearchResult>, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(self.timeout))
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(&json!({"query": query, "count": count}))
            .send()
            .await
            .map_err(|e| format!("DeepSeek search request failed: {e}"))?;
        let status = resp.status().as_u16();
        let body: Value = resp.json().await.unwrap_or(Value::Null);
        if status != 200 {
            let hint = match status {
                401 => "missing or invalid API key",
                402 => "insufficient DeepSeek account balance",
                408 => "request timed out; retry later",
                429 => "rate limit exceeded; retry later",
                504 => "upstream timed out; retry later",
                _ => "",
            };
            let detail = body
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("");
            let parts: Vec<&str> = [hint, detail]
                .into_iter()
                .filter(|p| !p.is_empty())
                .collect();
            let suffix = if parts.is_empty() {
                String::new()
            } else {
                format!(": {}", parts.join("; "))
            };
            return Err(format!("DeepSeek search failed (HTTP {status}){suffix}"));
        }
        let items = body
            .get("results")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(items
            .iter()
            .filter(|i| i.is_object())
            .map(|i| {
                let t = |k: &str| {
                    i.get(k)
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string()
                };
                let opt = |k: &str| Some(t(k)).filter(|s| !s.is_empty());
                SearchResult {
                    title: t("title"),
                    url: t("url"),
                    summary: opt("summary").unwrap_or_else(|| t("snippet")),
                    site_name: opt("site_name"),
                    date: opt("date"),
                }
            })
            .collect())
    }
}

/// The DeepSeek search key: `[web_search] api_key`, else the DeepSeek model
/// provider's key (`[llm] api_key` only counts while a DeepSeek model is
/// active — otherwise it belongs to another provider).
pub fn resolve_search_key(config: &Config) -> Option<String> {
    if let Some(k) = config.web_search.api_key.clone().filter(|k| !k.is_empty()) {
        return Some(k);
    }
    let active = resolve_model(&config.llm.model);
    if active.provider.id == "deepseek" {
        return resolve_api_key(&active, config);
    }
    let mut scoped = config.clone();
    scoped.llm.api_key = None;
    resolve_api_key(&resolve_model("deepseek-v4-pro"), &scoped)
}

/// The configured provider, or None (disabled, unknown, or no key).
pub fn build_provider(config: &Config) -> Option<DeepSeekSearch> {
    let s = &config.web_search;
    if !s.enabled || s.provider != "deepseek" {
        return None;
    }
    let key = resolve_search_key(config)?;
    let base = s
        .base_url
        .clone()
        .unwrap_or_else(|| DEEPSEEK_BASE_URL.into());
    Some(DeepSeekSearch {
        api_key: key,
        url: format!("{}/tools/search", base.trim_end_matches('/')),
        timeout: s.timeout,
    })
}

pub fn format_results(results: &[SearchResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut lines = vec![
                format!(
                    "[{}] {}",
                    i + 1,
                    if r.title.is_empty() { &r.url } else { &r.title }
                ),
                r.url.clone(),
            ];
            let meta: Vec<&str> = [r.site_name.as_deref(), r.date.as_deref()]
                .into_iter()
                .flatten()
                .collect();
            if !meta.is_empty() {
                lines.push(meta.join(" · "));
            }
            if !r.summary.is_empty() {
                lines.push(r.summary.clone());
            }
            lines.join("\n")
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub struct WebSearchTool {
    provider: DeepSeekSearch,
    default_count: u32,
}

impl WebSearchTool {
    pub fn new(provider: DeepSeekSearch, default_count: u32) -> Self {
        Self {
            provider,
            default_count: default_count.clamp(1, DeepSeekSearch::MAX_COUNT),
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &'static str {
        "web_search"
    }
    fn description(&self) -> String {
        format!(
            "Search the web for up-to-date information (docs, releases, news, error messages). Returns ranked \
             results with title, URL, date and a summary. Queries are at most {MAX_QUERY_CHARS} characters.{}",
            DeepSeekSearch::QUERY_HELP
        )
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "count": {"type": "integer", "description": format!("Number of results (1-{}, default {}).", DeepSeekSearch::MAX_COUNT, self.default_count)},
            },
            "required": ["query"],
        })
    }
    fn effect(&self) -> crate::effect::Effect {
        crate::effect::Effect::Fetch
    }
    async fn run(&self, args: Value, _ctx: &ToolContext) -> ToolResult {
        let Some(query) = arg_str(&args, "query")
            .map(str::trim)
            .filter(|q| !q.is_empty())
        else {
            return ToolResult::err("query must be a non-empty string");
        };
        let n = query.chars().count();
        if n > MAX_QUERY_CHARS {
            return ToolResult::err(format!(
                "query is {n} characters; the limit is {MAX_QUERY_CHARS}"
            ));
        }
        let count = match args.get("count") {
            None | Some(Value::Null) => self.default_count,
            Some(Value::Bool(_)) => return ToolResult::err("count must be an integer"),
            Some(_) => match arg_i64(&args, "count") {
                Some(c) => c.clamp(1, DeepSeekSearch::MAX_COUNT as i64) as u32,
                None => return ToolResult::err("count must be an integer"),
            },
        };
        match self.provider.search(query, count).await {
            Ok(r) if r.is_empty() => ToolResult::ok(format!("No results for: {query}")),
            Ok(r) => ToolResult::ok(truncate_output(format_results(&r), MAX_OUTPUT_BYTES)),
            Err(e) => ToolResult::err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        let out = format_results(&[SearchResult {
            title: "Rust".into(),
            url: "https://rust-lang.org".into(),
            summary: "A language".into(),
            site_name: Some("rust-lang".into()),
            date: None,
        }]);
        assert_eq!(
            out,
            "[1] Rust\nhttps://rust-lang.org\nrust-lang\nA language"
        );
    }

    #[test]
    fn key_scoping() {
        let mut cfg = Config::default();
        cfg.llm.model = "kimi-k2-0905-preview".into();
        cfg.llm.api_key = Some("moonshot-key".into());
        // [llm] api_key belongs to another provider: never sent to DeepSeek
        let k = resolve_search_key(&cfg);
        assert_ne!(k.as_deref(), Some("moonshot-key"));
        cfg.web_search.api_key = Some("ds".into());
        assert_eq!(resolve_search_key(&cfg).as_deref(), Some("ds"));
    }
}
