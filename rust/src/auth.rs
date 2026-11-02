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

fn read_raw(path: &std::path::Path) -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Write with 0600 permissions (keys are secrets), atomically.
fn write_raw(
    path: &std::path::Path,
    data: &serde_json::Map<String, serde_json::Value>,
) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("json.tmp");
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)?;
    let mut text = serde_json::to_string_pretty(data)?;
    text.push('\n');
    f.write_all(text.as_bytes())?;
    drop(f);
    std::fs::rename(&tmp, path)
}

/// Store (or replace) one provider's key.
pub fn save(provider: &str, key: &str, base_url: Option<&str>) -> std::io::Result<()> {
    save_at(&auth_path(), provider, key, base_url)
}

fn save_at(
    path: &std::path::Path,
    provider: &str,
    key: &str,
    base_url: Option<&str>,
) -> std::io::Result<()> {
    let mut data = read_raw(path);
    let mut entry = serde_json::json!({"type": "api", "key": key.trim()});
    if let Some(u) = base_url.filter(|u| !u.trim().is_empty()) {
        entry["base_url"] = serde_json::json!(u.trim());
    }
    data.insert(provider.to_string(), entry);
    write_raw(path, &data)
}

/// Delete a stored key; false when there was none.
pub fn remove(provider: &str) -> std::io::Result<bool> {
    remove_at(&auth_path(), provider)
}

fn remove_at(path: &std::path::Path, provider: &str) -> std::io::Result<bool> {
    let mut data = read_raw(path);
    if data.remove(provider).is_none() {
        return Ok(false);
    }
    write_raw(path, &data)?;
    Ok(true)
}

/// `sk-ab…wxyz`: enough to tell keys apart, never enough to use one.
pub fn mask(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 10 {
        return "•".repeat(chars.len());
    }
    let head: String = chars[..5].iter().collect();
    let tail: String = chars[chars.len() - 4..].iter().collect();
    format!("{head}…{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_list_remove_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("auth.json");
        save_at(&file, "glm", " sk-1234567890abcd ", None).unwrap();
        assert_eq!(read_raw(&file)["glm"]["key"], "sk-1234567890abcd");
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&file).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(remove_at(&file, "glm").unwrap());
        assert!(!remove_at(&file, "glm").unwrap());
        assert_eq!(mask("sk-1234567890abcd"), "sk-12…abcd");
        assert_eq!(mask("short"), "•••••");
    }
}
