//! Configuration: `~/.limbo/config.toml`, the same file the Python version reads.
//!
//! Every section and field has a default, so a missing file — or a file that
//! only sets `[llm] model` — just works. Sections the Rust version does not
//! use (e.g. `[daily]`, `[goal]`) are ignored instead of rejected.
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
    pub enabled: bool,
    /// Max tool-call iterations of one subagent run.
    pub max_iterations: u32,
    /// Model for subagents (empty = the main model).
    pub cheap_model: String,
}

impl Default for SubagentSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            max_iterations: 30,
            cheap_model: String::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub llm: LlmConfig,
    pub ui: UiConfig,
    pub tools: ToolsConfig,
    pub compaction: CompactionSettings,
    pub subagents: SubagentSettings,
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
