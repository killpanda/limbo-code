//! Configuration: `~/.limbo/config.toml`, the same file the Python version reads.
//!
//! Every section and field has a default, so a missing file — or a file that
//! only sets `[llm] model` — just works. Unknown keys are ignored instead of
//! rejected, so a config written for a newer Python release still loads.
//!
//! 🦀 `#[serde(default)]` on a struct means "fill any missing field from
//! `Default::default()`". We implement `Default` by hand where the defaults
//! are not zero/empty.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";

/// `~/.limbo`, or `$LIMBO_HOME` when set (handy for tests).
pub fn limbo_home() -> PathBuf {
    if let Ok(dir) = std::env::var("LIMBO_HOME") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".limbo")
}

pub fn default_config_path() -> PathBuf {
    limbo_home().join("config.toml")
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub temperature: f64,
    pub max_iterations: u32,
    pub thinking_effort: Option<String>,
    pub max_tokens: Option<u32>,
    pub max_retries: u32,
    pub retry_base_delay: f64,
    pub timeout: f64,
    pub connect_timeout: f64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: DEFAULT_BASE_URL.into(),
            model: "deepseek-v4-pro".into(),
            temperature: 0.2,
            max_iterations: 50,
            thinking_effort: None,
            max_tokens: None,
            max_retries: 3,
            retry_base_delay: 1.0,
            timeout: 600.0,
            connect_timeout: 30.0,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ProviderOverride {
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    /// `neon` (default), `aurora` or `ember`.
    pub theme: Option<String>,
    pub show_banner: bool,
    /// `full` / `reduced` / `off` — how much animation the TUI plays.
    pub motion: Option<String>,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: None,
            show_banner: true,
            motion: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    pub bash_enabled: bool,
    pub parallel: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            bash_enabled: true,
            parallel: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CompactionSettings {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SubagentSettings {
    /// false: no supervisor at all (no board, no /agents, no task tools).
    pub enabled: bool,
    /// off: no task tools for the model; suggest: the model's plan waits
    /// for the user's approval; on: it runs at once.
    pub auto: String,
    pub max_parallel: usize,
    /// Live (unfinished) tasks at a time.
    pub max_tasks: usize,
    /// Per-task budget of fresh tokens (0 = unlimited).
    pub task_token_budget: u64,
    /// Model for profiles with `model: cheap` (empty = the main model).
    pub cheap_model: String,
    /// Which notices wake an idle main agent.
    pub wake_main_on: Vec<String>,
    pub on_dep_fail: String,
    /// Tasks that were running when the process stopped: ask / auto / pause.
    pub on_resume: String,
    pub default_isolation: String,
    /// Characters of a finished task's summary delivered to the main agent.
    pub result_chars: usize,
    /// Auto checkpoint every N tool calls of a subagent (0 = off).
    pub auto_checkpoint_every: usize,
    /// Max tool-call iterations of one subagent run (Rust-only knob).
    pub max_iterations: u32,
}

impl Default for SubagentSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            auto: "suggest".into(),
            max_parallel: 3,
            max_tasks: 12,
            task_token_budget: 400_000,
            cheap_model: String::new(),
            wake_main_on: vec!["question".into(), "alert".into(), "wave_done".into()],
            on_dep_fail: "notify".into(),
            on_resume: "ask".into(),
            default_isolation: "shared".into(),
            result_chars: 2000,
            auto_checkpoint_every: 8,
            max_iterations: 50,
        }
    }
}

/// `[web_search]`: the `web_search` tool.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WebSearchSettings {
    pub enabled: bool,
    /// Backend id (only "deepseek" for now).
    pub provider: String,
    /// Unset: inherit the matching model provider's key.
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub count: u32,
    pub timeout: f64,
}

impl Default for WebSearchSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            provider: "deepseek".into(),
            api_key: None,
            base_url: None,
            count: 10,
            timeout: 30.0,
        }
    }
}

/// `[daily]`: the daily (non-coding) session posture.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DailySettings {
    /// Workdir of `limbo daily` sessions.
    pub workdir: String,
    /// Model for `limbo daily` (empty = `[llm] model`).
    pub model: String,
    /// Persistent memory (`~/.limbo/memory/`).
    pub memory: bool,
    pub user_chars: usize,
    pub notes_chars: usize,
    /// Remind the model about memory after this many quiet turns (0 = never).
    pub memory_nudge_turns: u32,
    /// Automatic skill review after this many tool rounds (0 = only /learn).
    #[serde(alias = "learn_after_tool_calls")]
    pub learn_after_rounds: u32,
    pub learn_cooldown_turns: u32,
    pub fetch_max_chars: usize,
    pub fetch_timeout: f64,
    /// TUI look while daily mode is on: "dawn" or "badge".
    pub look: String,
}

impl Default for DailySettings {
    fn default() -> Self {
        Self {
            workdir: "~/limbo-daily".into(),
            model: String::new(),
            memory: true,
            user_chars: 1500,
            notes_chars: 2500,
            memory_nudge_turns: 10,
            learn_after_rounds: 10,
            learn_cooldown_turns: 3,
            fetch_max_chars: 20_000,
            fetch_timeout: 30.0,
            look: "dawn".into(),
        }
    }
}

/// `[goal]`: closed-loop /goal mode.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GoalSettings {
    pub max_rounds: u32,
    pub verify_timeout_ms: u64,
}

impl Default for GoalSettings {
    fn default() -> Self {
        Self {
            max_rounds: 10,
            verify_timeout_ms: 600_000,
        }
    }
}

/// `[queue]`: what happens to queued messages when a turn fails.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QueueSettings {
    /// "continue" | "keep" | "withdraw".
    pub after_failed_run: String,
}

impl Default for QueueSettings {
    fn default() -> Self {
        Self {
            after_failed_run: "continue".into(),
        }
    }
}

/// `[gateway]`: `limbo gateway`, the chat front end (Telegram / WeChat).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GatewaySettings {
    /// Workdir of chat sessions.
    pub workdir: String,
    /// Capability set of new chat sessions: "daily" or "coding".
    pub posture: String,
    /// Model for chat sessions (empty = `[llm] model`).
    pub model: String,
    /// Unload an idle chat's agent after this many seconds (0 = never).
    pub idle_unload_secs: u64,
    /// Largest attachment downloaded from a chat, in MB.
    pub max_attachment_mb: u64,
    /// Show tool progress lines in chats that can edit messages.
    pub show_tools: bool,
}

impl Default for GatewaySettings {
    fn default() -> Self {
        Self {
            workdir: "~/limbo-chat".into(),
            posture: "daily".into(),
            model: String::new(),
            idle_unload_secs: 1800,
            max_attachment_mb: 20,
            show_tools: true,
        }
    }
}

/// Ids in config files may be written as numbers (`owners = [12345]`) or
/// strings; both become strings.
fn de_ids<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    let raw: Vec<toml::Value> = Deserialize::deserialize(d)?;
    Ok(raw
        .into_iter()
        .filter_map(|v| match v {
            toml::Value::Integer(i) => Some(i.to_string()),
            toml::Value::String(s) => Some(s),
            _ => None,
        })
        .collect())
}

/// `[channels.telegram]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TelegramSettings {
    pub enabled: bool,
    /// Telegram user ids allowed to talk to the bot (besides the one
    /// recorded by `limbo channels login telegram`).
    #[serde(deserialize_with = "de_ids")]
    pub owners: Vec<String>,
    /// HTTP(S)/SOCKS proxy for api.telegram.org.
    pub proxy: String,
    pub api_base: String,
    /// Groups: "off" (ignore) or "mention" (answer @mentions and replies).
    pub groups: String,
}

impl Default for TelegramSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            owners: Vec::new(),
            proxy: String::new(),
            api_base: crate::channels::telegram::DEFAULT_API_BASE.into(),
            groups: "mention".into(),
        }
    }
}

/// `[channels.weixin]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WeixinSettings {
    pub enabled: bool,
    /// iLink user ids allowed besides whoever scanned the login QR code.
    #[serde(deserialize_with = "de_ids")]
    pub owners: Vec<String>,
    /// `iLink-App-Id` header.
    pub app_id: String,
    /// `iLink-App-ClientVersion` (the protocol version the server expects).
    pub client_version: String,
}

impl Default for WeixinSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            owners: Vec::new(),
            app_id: "bot".into(),
            client_version: "2.4.10".into(),
        }
    }
}

/// `[channels]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ChannelsSettings {
    pub telegram: TelegramSettings,
    pub weixin: WeixinSettings,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub llm: LlmConfig,
    pub ui: UiConfig,
    pub tools: ToolsConfig,
    pub web_search: WebSearchSettings,
    pub compaction: CompactionSettings,
    pub goal: GoalSettings,
    pub queue: QueueSettings,
    pub subagents: SubagentSettings,
    pub daily: DailySettings,
    pub gateway: GatewaySettings,
    pub channels: ChannelsSettings,
    pub providers: HashMap<String, ProviderOverride>,
}

impl Config {
    /// Load the config, falling back to defaults (with a warning on stderr)
    /// when the file is missing or malformed — a typo must never stop Limbo
    /// from starting.
    pub fn load(path: Option<&Path>) -> (Config, Vec<String>) {
        let path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(default_config_path);
        let mut warnings = Vec::new();
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(_) => return (Config::default(), warnings),
        };
        match toml::from_str::<Config>(&text) {
            Ok(mut cfg) => {
                if cfg.llm.max_iterations == 0 {
                    cfg.llm.max_iterations = 1;
                }
                if cfg.compaction.reserve_tokens < 1024 {
                    warnings.push("reserve_tokens is too small; reset to 16384".into());
                    cfg.compaction.reserve_tokens = 16_384;
                }
                if cfg.daily.look != "dawn" && cfg.daily.look != "badge" {
                    warnings.push(format!(
                        "daily look={:?} is unknown; reset to 'dawn'",
                        cfg.daily.look
                    ));
                    cfg.daily.look = "dawn".into();
                }
                if cfg.web_search.count < 1 {
                    cfg.web_search.count = 10;
                }
                if !matches!(
                    cfg.queue.after_failed_run.as_str(),
                    "continue" | "keep" | "withdraw"
                ) {
                    cfg.queue.after_failed_run = "continue".into();
                }
                for (field, value, default) in [
                    ("user_chars", &mut cfg.daily.user_chars, 1500),
                    ("notes_chars", &mut cfg.daily.notes_chars, 2500),
                    ("fetch_max_chars", &mut cfg.daily.fetch_max_chars, 20_000),
                ] {
                    if *value < 100 {
                        warnings.push(format!(
                            "daily {field}={value} is too small; reset to {default}"
                        ));
                        *value = default;
                    }
                }
                if cfg.compaction.keep_recent_tokens < 1000 {
                    warnings.push("keep_recent_tokens is too small; reset to 20000".into());
                    cfg.compaction.keep_recent_tokens = 20_000;
                }
                (cfg, warnings)
            }
            Err(e) => {
                warnings.push(format!(
                    "Invalid config file {}: {e}. Using defaults.",
                    path.display()
                ));
                (Config::default(), warnings)
            }
        }
    }
}

/// Persist `[llm] model` (and optionally `thinking_effort`) to config.toml,
/// keeping comments and formatting — `toml_edit` edits the document tree
/// instead of re-serializing it.
pub fn save_model_to_config(model: &str, path: Option<&Path>) -> anyhow::Result<()> {
    let path = path
        .map(Path::to_path_buf)
        .unwrap_or_else(default_config_path);
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    if !doc.contains_table("llm") {
        doc["llm"] = toml_edit::table();
    }
    doc["llm"]["model"] = toml_edit::value(model);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, doc.to_string())?;
    Ok(())
}

/// Persist `[llm] thinking_effort` (None removes it).
pub fn save_thinking_to_config(level: Option<&str>, path: Option<&Path>) -> anyhow::Result<()> {
    let path = path
        .map(Path::to_path_buf)
        .unwrap_or_else(default_config_path);
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let mut doc: toml_edit::DocumentMut = text.parse()?;
    if !doc.contains_table("llm") {
        doc["llm"] = toml_edit::table();
    }
    match level {
        Some(l) => doc["llm"]["thinking_effort"] = toml_edit::value(l),
        None => {
            if let Some(t) = doc["llm"].as_table_mut() {
                t.remove("thinking_effort");
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, doc.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_fill_missing_sections() {
        let cfg: Config =
            toml::from_str("[llm]\nmodel = \"glm-4.7\"\n[daily]\nmemory = true\n").unwrap();
        assert_eq!(cfg.llm.model, "glm-4.7");
        assert_eq!(cfg.llm.max_iterations, 50);
        assert!(cfg.tools.bash_enabled);
        assert_eq!(cfg.compaction.reserve_tokens, 16_384);
    }

    #[test]
    fn provider_overrides_parse() {
        let cfg: Config =
            toml::from_str("[providers.glm]\nbase_url = \"https://x\"\nheaders = { A = \"b\" }\n")
                .unwrap();
        assert_eq!(cfg.providers["glm"].base_url.as_deref(), Some("https://x"));
        assert_eq!(cfg.providers["glm"].headers["A"], "b");
    }

    #[test]
    fn save_thinking_sets_and_removes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        std::fs::write(&p, "# keep\n[llm]\nmodel = \"x\"\n").unwrap();
        save_thinking_to_config(Some("high"), Some(&p)).unwrap();
        let t = std::fs::read_to_string(&p).unwrap();
        assert!(t.contains("# keep") && t.contains("thinking_effort = \"high\""));
        save_thinking_to_config(None, Some(&p)).unwrap();
        assert!(
            !std::fs::read_to_string(&p)
                .unwrap()
                .contains("thinking_effort")
        );
    }

    #[test]
    fn save_model_keeps_comments() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("config.toml");
        std::fs::write(&p, "# my comment\n[llm]\nmodel = \"a\"\n").unwrap();
        save_model_to_config("kimi-k3", Some(&p)).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("# my comment"));
        assert!(text.contains("model = \"kimi-k3\""));
    }
}
