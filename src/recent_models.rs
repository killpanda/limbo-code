//! Recently used models (`~/.limbo/recent_models.json`), newest first.
//! The model picker lists them on top.

use std::path::{Path, PathBuf};

use crate::config::limbo_home;

pub const LIMIT: usize = 5;

fn path() -> PathBuf {
    limbo_home().join("recent_models.json")
}

pub fn load() -> Vec<String> {
    load_at(&path())
}

fn load_at(p: &Path) -> Vec<String> {
    std::fs::read_to_string(p)
        .ok()
        .and_then(|t| serde_json::from_str::<Vec<serde_json::Value>>(&t).ok())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .take(LIMIT)
        .collect()
}

pub fn push(model: &str) {
    push_at(&path(), model);
}

fn push_at(p: &Path, model: &str) {
    let mut models = vec![model.to_string()];
    models.extend(load_at(p).into_iter().filter(|m| m != model));
    models.truncate(LIMIT);
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(p, serde_json::to_string(&models).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newest_first_deduplicated_and_capped() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("r.json");
        for m in ["a", "b", "c", "d", "e", "f", "b"] {
            push_at(&p, m);
        }
        assert_eq!(load_at(&p), ["b", "f", "e", "d", "c"]);
    }
}
