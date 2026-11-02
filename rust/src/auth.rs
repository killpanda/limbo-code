//! Stored provider credentials (`~/.limbo/auth.json`), shared with the
//! Python version: `{"kimi-coding": {"type": "api", "key": "sk-...", "base_url": null}}`.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::limbo_home;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credential {
    pub key: String,
    #[serde(default)]
    pub base_url: Option<String>,
}

pub fn auth_path() -> PathBuf {
    std::env::var("LIMBO_AUTH_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| limbo_home().join("auth.json"))
}

pub fn load_all() -> HashMap<String, Credential> {
    let Ok(text) = std::fs::read_to_string(auth_path()) else {
        return HashMap::new();
    };
    let raw: HashMap<String, serde_json::Value> = serde_json::from_str(&text).unwrap_or_default();
    raw.into_iter()
        .filter_map(|(k, v)| {
            serde_json::from_value::<Credential>(v)
                .ok()
                .filter(|c| !c.key.is_empty())
                .map(|c| (k, c))
        })
        .collect()
}

pub fn get(provider: &str) -> Option<Credential> {
    load_all().remove(provider)
}
