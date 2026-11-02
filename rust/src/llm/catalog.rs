//! Provider/model catalog: which API dialect, endpoint, key and quirks
//! each model uses. Mirrors `src/limbo/llm/catalog.py`.
//!
//! A *provider* owns the dialect + endpoint + credential env var; a *model*
//! carries its own context window, output cap and thinking format. Unknown
//! models fall back to a generic OpenAI-compatible provider, so any
//! compatible endpoint works through `[llm] base_url` + `model`.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use serde_json::{Value, json};

use crate::auth;
use crate::config::{Config, DEFAULT_BASE_URL};

pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
pub const DEFAULT_MAX_TOKENS: u32 = 16_384;

/// The wire format a provider speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAiCompletions,
    AnthropicMessages,
    OpenAiResponses,
}

/// How a reasoning model's thinking is switched/tuned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingFormat {
    /// `reasoning_effort` parameter.
    OpenAi,
    /// `thinking: {type: enabled|disabled}`.
    DeepSeek,
    /// DeepSeek shape plus `clear_thinking: false`.
    Zai,
    /// `reasoning: {effort, summary}` on the Responses API.
    OpenAiResponses,
    /// Anthropic adaptive thinking + `output_config.effort`.
    AnthropicAdaptive,
}

#[derive(Debug, Clone)]
pub struct ProviderSpec {
    pub id: &'static str,
    pub dialect: Dialect,
    pub base_url: &'static str,
    pub api_key_env: Option<&'static str>,
    pub headers: Vec<(&'static str, &'static str)>,
    /// Extra body fields sent only when the request carries tools.
    pub tool_extra_body: Option<Value>,
    pub default_api_key: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub id: String,
    pub provider: ProviderSpec,
    pub context_window: u64,
    pub max_tokens: u32,
    pub reasoning: bool,
    pub vision: bool,
    pub thinking_format: Option<ThinkingFormat>,
    /// Supported thinking levels → provider value (ordered for display).
    pub thinking_levels: Vec<(&'static str, &'static str)>,
    pub thinking_can_disable: bool,
    pub requires_reasoning_content: bool,
    /// Wire model name when it differs from `id` (Magpie).
    pub api_model: Option<String>,
}

impl ModelSpec {
    pub fn wire_id(&self) -> &str {
        self.api_model.as_deref().unwrap_or(&self.id)
    }
    pub fn map_effort(&self, effort: &str) -> Option<&'static str> {
        self.thinking_levels
            .iter()
            .find(|(k, _)| *k == effort)
            .map(|(_, v)| *v)
    }
    fn generic(id: &str, provider: ProviderSpec) -> Self {
        Self {
            id: id.to_string(),
            provider,
            context_window: DEFAULT_CONTEXT_WINDOW,
            max_tokens: DEFAULT_MAX_TOKENS,
            reasoning: false,
            vision: false,
            thinking_format: None,
            thinking_levels: vec![],
            thinking_can_disable: true,
            requires_reasoning_content: false,
            api_model: None,
        }
    }
}

// -- providers --------------------------------------------------------------

fn deepseek() -> ProviderSpec {
    ProviderSpec {
        id: "deepseek",
        dialect: Dialect::OpenAiCompletions,
        base_url: DEFAULT_BASE_URL,
        api_key_env: Some("DEEPSEEK_API_KEY"),
        headers: vec![],
        tool_extra_body: None,
        default_api_key: None,
    }
}
fn moonshot() -> ProviderSpec {
    ProviderSpec {
        id: "moonshotai",
        base_url: "https://api.moonshot.ai/v1",
        api_key_env: Some("MOONSHOT_API_KEY"),
        ..deepseek()
    }
}
fn generic_openai() -> ProviderSpec {
    ProviderSpec {
        id: "custom",
        base_url: "",
        api_key_env: None,
        ..deepseek()
    }
}
fn kimi_coding() -> ProviderSpec {
    ProviderSpec {
        id: "kimi-coding",
        dialect: Dialect::AnthropicMessages,
        base_url: "https://api.kimi.com/coding",
        api_key_env: Some("KIMI_API_KEY"),
        headers: vec![("User-Agent", "KimiCLI/1.5")],
        tool_extra_body: None,
        default_api_key: None,
    }
}
fn glm_coding() -> ProviderSpec {
    ProviderSpec {
        id: "glm",
        base_url: "https://open.bigmodel.cn/api/coding/paas/v4",
        api_key_env: Some("ZHIPUAI_API_KEY"),
        tool_extra_body: Some(json!({"tool_stream": true})),
        ..deepseek()
    }
}
fn codex() -> ProviderSpec {
    ProviderSpec {
        id: "codex",
        dialect: Dialect::OpenAiResponses,
        base_url: "https://api.openai.com/v1",
        api_key_env: Some("CODEX_API_KEY"),
        headers: vec![],
        tool_extra_body: None,
        default_api_key: None,
    }
}
fn magpie() -> ProviderSpec {
    ProviderSpec {
        id: "magpie",
        base_url: "http://127.0.0.1:3425/v1",
        api_key_env: Some("MAGPIE_API_KEY"),
        default_api_key: Some("magpie"),
        ..deepseek()
    }
}

pub const MAGPIE_PREFIX: &str = "magpie/";

// -- model helpers ----------------------------------------------------------

fn ds(id: &str, vision: bool) -> ModelSpec {
    ModelSpec {
        context_window: 1_000_000,
        max_tokens: 65_536,
        reasoning: true,
        vision,
        thinking_format: Some(ThinkingFormat::DeepSeek),
        ..ModelSpec::generic(id, deepseek())
    }
}

fn moon(id: &str) -> ModelSpec {
    ModelSpec {
        context_window: 262_144,
        max_tokens: 262_144,
        vision: true,
        ..ModelSpec::generic(id, moonshot())
    }
}

fn moon_thinking(id: &str, can_disable: bool) -> ModelSpec {
    ModelSpec {
        reasoning: true,
        thinking_format: Some(ThinkingFormat::DeepSeek),
        thinking_can_disable: can_disable,
        ..moon(id)
    }
}

fn kimi_c(id: &str, context_window: u64, max_tokens: u32) -> ModelSpec {
    ModelSpec {
        context_window,
        max_tokens,
        reasoning: true,
        vision: true,
        thinking_format: Some(ThinkingFormat::AnthropicAdaptive),
        thinking_levels: vec![("low", "low"), ("high", "high"), ("max", "max")],
        thinking_can_disable: false,
        ..ModelSpec::generic(id, kimi_coding())
    }
}

fn glm(id: &str, context_window: u64, max_tokens: u32, vision: bool) -> ModelSpec {
    ModelSpec {
        context_window,
        max_tokens,
        reasoning: true,
        vision,
        thinking_format: Some(ThinkingFormat::Zai),
        ..ModelSpec::generic(id, glm_coding())
    }
}

fn codex_m(id: &str, context_window: u64, vision: bool, supports_max: bool) -> ModelSpec {
    let mut levels = vec![
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
    ];
    if supports_max {
        levels.push(("max", "max"));
    }
    ModelSpec {
        context_window,
        max_tokens: 128_000,
        reasoning: true,
        vision,
        thinking_format: Some(ThinkingFormat::OpenAiResponses),
        thinking_levels: levels,
        ..ModelSpec::generic(id, codex())
    }
}

/// The built-in catalog, built once on first use.
///
/// 🦀 `OnceLock` is a thread-safe "initialize on first access" cell — the
/// Rust standard library's answer to a lazily-built global.
pub fn catalog() -> &'static BTreeMap<String, ModelSpec> {
    static CATALOG: OnceLock<BTreeMap<String, ModelSpec>> = OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut m = vec![
            ds("deepseek-flash", true),
            ds("deepseek-v4-pro", false),
            ds("deepseek-v4-flash", false),
            ds("deepseek-v4-flash-vision-exp", true),
            ModelSpec {
                context_window: 131_072,
                max_tokens: 16_384,
                ..moon("kimi-k2-0711-preview")
            },
            moon("kimi-k2-0905-preview"),
            moon("kimi-k2-turbo-preview"),
            moon_thinking("kimi-k2-thinking", true),
            moon_thinking("kimi-k2-thinking-turbo", true),
            moon_thinking("kimi-k2.5", true),
            moon_thinking("kimi-k2.6", true),
            moon_thinking("kimi-k2.7-code", false),
            moon_thinking("kimi-k2.7-code-highspeed", false),
            ModelSpec {
                context_window: 1_048_576,
                max_tokens: 131_072,
                reasoning: true,
                thinking_format: Some(ThinkingFormat::OpenAi),
                thinking_levels: vec![("low", "low"), ("high", "high"), ("max", "max")],
                thinking_can_disable: false,
                requires_reasoning_content: true,
                ..moon("kimi-k3")
            },
            kimi_c("k3", 1_048_576, 131_072),
            kimi_c("k3-256k", 262_144, 32_768),
            kimi_c("kimi-for-coding", 1_048_576, 131_072),
            kimi_c("kimi-for-coding-highspeed", 262_144, 32_768),
            glm("glm-4.5-air", 131_072, 98_304, false),
            glm("glm-4.7", 204_800, 131_072, false),
            glm("glm-5-turbo", 200_000, 131_072, false),
            glm("glm-5.1", 200_000, 131_072, false),
            ModelSpec {
                thinking_levels: vec![("low", "high"), ("high", "high"), ("max", "max")],
                ..glm("glm-5.2", 1_000_000, 131_072, false)
            },
            glm("glm-5v-turbo", 200_000, 131_072, true),
            codex_m("gpt-5.5", 1_000_000, true, false),
            codex_m("gpt-5.3-codex-spark", 128_000, false, false),
            codex_m("gpt-5.4", 272_000, true, false),
            codex_m("gpt-5.4-mini", 272_000, true, false),
            codex_m("gpt-5.6-sol", 272_000, true, true),
            codex_m("gpt-5.6-terra", 272_000, true, true),
            codex_m("gpt-5.6-luna", 272_000, true, true),
        ];
        m.push(magpie_model("magpie/deepseek/deepseek-flash"));
        m.into_iter().map(|s| (s.id.clone(), s)).collect()
    })
}

/// `magpie/<upstream>/<model>`: metadata guessed from the catalog entry with
/// the same bare model name, thinking through `reasoning_effort`.
fn magpie_model(model_id: &str) -> ModelSpec {
    let wire = model_id.trim_start_matches(MAGPIE_PREFIX).to_string();
    let bare = wire.rsplit('/').next().unwrap_or(&wire).to_string();
    let mut spec = ModelSpec {
        api_model: Some(wire),
        ..ModelSpec::generic(model_id, magpie())
    };
    // `catalog_peek` builds the base spec without touching the global
    // catalog, which may itself be mid-construction here.
    if let Some(b) = catalog_peek(&bare) {
        spec.context_window = b.context_window;
        spec.max_tokens = b.max_tokens;
        spec.reasoning = b.reasoning;
        spec.vision = b.vision;
        spec.requires_reasoning_content = b.requires_reasoning_content;
        if b.reasoning {
            spec.thinking_format = Some(ThinkingFormat::OpenAi);
            spec.thinking_levels = if b.thinking_levels.is_empty() {
                vec![("low", "low"), ("medium", "medium"), ("high", "high")]
            } else {
                b.thinking_levels.iter().map(|(k, _)| (*k, *k)).collect()
            };
            spec.thinking_can_disable = false;
        }
    }
    spec
}

fn catalog_peek(id: &str) -> Option<ModelSpec> {
    // Build the base spec without touching the global catalog.
    Some(match id {
        "deepseek-flash" => ds(id, true),
        "deepseek-v4-pro" | "deepseek-v4-flash" => ds(id, false),
        "kimi-k3" => ModelSpec {
            context_window: 1_048_576,
            max_tokens: 131_072,
            reasoning: true,
            thinking_levels: vec![("low", "low"), ("high", "high"), ("max", "max")],
            requires_reasoning_content: true,
            ..moon(id)
        },
        "glm-4.7" => glm(id, 204_800, 131_072, false),
        "glm-5.1" => glm(id, 200_000, 131_072, false),
        "glm-5.2" => glm(id, 1_000_000, 131_072, false),
        "gpt-5.5" => codex_m(id, 1_000_000, true, false),
        "gpt-5.4" => codex_m(id, 272_000, true, false),
        _ => return None,
    })
}

/// Look up a model; unknown ids get a generic OpenAI-compatible spec.
pub fn resolve_model(model_id: &str) -> ModelSpec {
    if model_id.starts_with(MAGPIE_PREFIX) {
        return magpie_model(model_id);
    }
    catalog()
        .get(model_id)
        .cloned()
        .unwrap_or_else(|| ModelSpec::generic(model_id, generic_openai()))
}

/// Values `[llm] thinking_effort` can take (None = provider default).
pub fn thinking_choices(spec: &ModelSpec) -> Vec<Option<String>> {
    let Some(fmt) = spec.thinking_format else {
        return vec![];
    };
    if !spec.reasoning {
        return vec![];
    }
    let mut out: Vec<Option<String>> = vec![None];
    out.extend(
        spec.thinking_levels
            .iter()
            .map(|(k, _)| Some(k.to_string())),
    );
    if matches!(fmt, ThinkingFormat::DeepSeek | ThinkingFormat::Zai) {
        if spec.thinking_levels.is_empty() {
            out.push(Some("on".into()));
        }
        if spec.thinking_can_disable {
            out.push(Some("off".into()));
        }
    }
    out
}

/// Split `kimi-k3:high` into the model and a thinking level.
pub fn split_model_arg(value: &str) -> (String, Option<String>) {
    const WORDS: &[&str] = &["off", "on", "low", "medium", "high", "xhigh", "max"];
    if let Some((model, level)) = value.rsplit_once(':') {
        let level_l = level.to_lowercase();
        if !model.is_empty() && WORDS.contains(&level_l.as_str()) && !catalog().contains_key(value)
        {
            return (model.to_string(), Some(level_l));
        }
    }
    (value.to_string(), None)
}

pub fn provider_display_name(id: &str) -> &str {
    match id {
        "deepseek" => "DeepSeek",
        "moonshotai" => "Moonshot",
        "kimi-coding" => "Kimi For Coding",
        "glm" => "GLM Coding Plan",
        "codex" => "OpenAI Codex",
        "magpie" => "Magpie（本地网关）",
        "custom" => "自定义 OpenAI 兼容",
        other => other,
    }
}

// -- resolution against config ----------------------------------------------

/// Effective base URL: `[providers.<id>]` → stored login → explicit `[llm]`
/// → the provider's built-in endpoint.
pub fn resolve_base_url(spec: &ModelSpec, config: &Config) -> String {
    let override_url = config
        .providers
        .get(spec.provider.id)
        .and_then(|o| o.base_url.clone());
    let stored = auth::get(spec.provider.id).and_then(|c| c.base_url);
    if let Some(url) = override_url.or(stored) {
        return url;
    }
    let configured = &config.llm.base_url;
    if configured != DEFAULT_BASE_URL || spec.provider.base_url.is_empty() {
        return configured.clone();
    }
    spec.provider.base_url.to_string()
}

pub fn resolve_api_key_env(spec: &ModelSpec, config: &Config) -> Option<String> {
    config
        .providers
        .get(spec.provider.id)
        .and_then(|o| o.api_key_env.clone())
        .or_else(|| spec.provider.api_key_env.map(str::to_string))
}

/// Effective API key: `[providers.<id>] api_key` → auth.json → `[llm]
/// api_key` → env var → provider placeholder.
pub fn resolve_api_key(spec: &ModelSpec, config: &Config) -> Option<String> {
    if let Some(k) = config
        .providers
        .get(spec.provider.id)
        .and_then(|o| o.api_key.clone())
    {
        return Some(k);
    }
    if let Some(c) = auth::get(spec.provider.id) {
        return Some(c.key);
    }
    if let Some(k) = config.llm.api_key.clone() {
        return Some(k);
    }
    if let Some(env) = resolve_api_key_env(spec, config)
        && let Ok(v) = std::env::var(env)
        && !v.is_empty()
    {
        return Some(v);
    }
    spec.provider.default_api_key.map(str::to_string)
}

pub fn resolve_headers(spec: &ModelSpec, config: &Config) -> HashMap<String, String> {
    let mut out: HashMap<String, String> = spec
        .provider
        .headers
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    if let Some(o) = config.providers.get(spec.provider.id) {
        out.extend(o.headers.clone());
    }
    out
}

pub fn missing_api_key_message(spec: &ModelSpec, config: &Config) -> String {
    let p = spec.provider.id;
    let mut lines = vec![
        format!("No API key for provider '{p}' (model '{}').", spec.id),
        "Set one of:".into(),
        format!("  [providers.{p}]"),
        "  api_key = \"your-key\"".into(),
        "  [llm]".into(),
        "  api_key = \"your-key\"".into(),
    ];
    if let Some(env) = resolve_api_key_env(spec, config) {
        lines.push(format!("  or set the ${env} env var"));
    }
    lines.push(format!("  or add it to {}", auth::auth_path().display()));
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_and_unknown_models() {
        let k3 = resolve_model("kimi-k3");
        assert_eq!(k3.provider.id, "moonshotai");
        assert!(k3.requires_reasoning_content);
        let unknown = resolve_model("my-local-llm");
        assert_eq!(unknown.provider.id, "custom");
        assert_eq!(unknown.context_window, DEFAULT_CONTEXT_WINDOW);
    }

    #[test]
    fn magpie_borrows_metadata() {
        let m = resolve_model("magpie/deepseek/deepseek-v4-pro");
        assert_eq!(m.wire_id(), "deepseek/deepseek-v4-pro");
        assert_eq!(m.context_window, 1_000_000);
        assert_eq!(m.thinking_format, Some(ThinkingFormat::OpenAi));
    }

    #[test]
    fn split_model_levels() {
        assert_eq!(
            split_model_arg("kimi-k3:high"),
            ("kimi-k3".into(), Some("high".into()))
        );
        assert_eq!(split_model_arg("qwen3:32b"), ("qwen3:32b".into(), None));
    }

    #[test]
    fn thinking_choices_for_deepseek() {
        let c = thinking_choices(&resolve_model("deepseek-v4-pro"));
        assert_eq!(c, vec![None, Some("on".into()), Some("off".into())]);
    }
}
