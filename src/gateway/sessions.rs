//! Which limbo session each chat is talking to: `~/.limbo/channels/sessions.json`,
//! `peer key → { current session, earlier sessions }`. A chat always has at
//! most one live session; `/new` starts another and keeps the old one in the
//! chat's history for `/resume`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::channels::PeerKey;

const HISTORY: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerEntry {
    pub peer: PeerKey,
    pub current: String,
    /// Earlier sessions of this chat, newest first.
    #[serde(default)]
    pub history: Vec<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SessionMap {
    #[serde(default)]
    pub peers: BTreeMap<String, PeerEntry>,
    #[serde(skip)]
    path: PathBuf,
}

impl SessionMap {
    pub fn load(path: &Path) -> Self {
        let mut m: SessionMap = std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        m.path = path.to_path_buf();
        m
    }

    fn save(&self) {
        if let Ok(text) = serde_json::to_string_pretty(self)
            && let Err(e) = crate::channels::store::write_private(&self.path, text.as_bytes())
        {
            eprintln!("warning: cannot save {}: {e}", self.path.display());
        }
    }

    pub fn current(&self, peer: &PeerKey) -> Option<String> {
        self.peers.get(&peer.key()).map(|e| e.current.clone())
    }

    /// Sessions this chat may `/resume` (current first).
    pub fn known(&self, peer: &PeerKey) -> Vec<String> {
        self.peers
            .get(&peer.key())
            .map(|e| {
                let mut v = vec![e.current.clone()];
                v.extend(e.history.iter().cloned());
                v
            })
            .unwrap_or_default()
    }

    /// Make `session` the chat's current session.
    pub fn set(&mut self, peer: &PeerKey, session: &str) {
        let entry = self.peers.entry(peer.key()).or_insert_with(|| PeerEntry {
            peer: peer.clone(),
            current: String::new(),
            history: vec![],
        });
        if entry.current == session {
            return;
        }
        let old = std::mem::replace(&mut entry.current, session.to_string());
        entry.history.retain(|s| s != session && s != &old);
        if !old.is_empty() {
            entry.history.insert(0, old);
        }
        entry.history.truncate(HISTORY);
        self.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::ChannelKind;

    #[test]
    fn current_and_history_survive_reload() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sessions.json");
        let peer = PeerKey {
            channel: ChannelKind::Telegram,
            account: "1".into(),
            chat: "2".into(),
            thread: None,
        };
        let mut m = SessionMap::load(&path);
        assert_eq!(m.current(&peer), None);
        m.set(&peer, "s1");
        m.set(&peer, "s2");
        m.set(&peer, "s1");
        let m = SessionMap::load(&path);
        assert_eq!(m.current(&peer).as_deref(), Some("s1"));
        assert_eq!(m.known(&peer), vec!["s1", "s2"]);
    }
}
