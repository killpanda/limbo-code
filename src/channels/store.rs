//! On-disk state of the chat gateway, under `~/.limbo/channels/`:
//!
//! - `credentials.json` (0600): bot tokens from `limbo channels login` and
//!   the owners recorded at login;
//! - `cursor-<channel>-<account>`: each adapter's long-poll cursor (Telegram
//!   `offset`, WeChat `get_updates_buf`);
//! - `gateway.lock`: one gateway per machine (a bot token allows one poller).
//!
//! 🦀 Rust notes
//! - `flock(2)` through `libc` works the same on Linux and macOS; the lock
//!   is released by the kernel when the process dies, so a crash never
//!   leaves a stale lock behind.

use std::fs::File;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::limbo_home;

pub fn channels_dir() -> PathBuf {
    limbo_home().join("channels")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelegramCred {
    pub token: String,
    #[serde(default)]
    pub bot_id: String,
    #[serde(default)]
    pub bot_username: String,
    /// User ids recorded by `limbo channels login telegram`.
    #[serde(default)]
    pub owners: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WeixinCred {
    pub token: String,
    /// `ilink_bot_id`.
    pub bot_id: String,
    /// API host the login was confirmed on.
    #[serde(default)]
    pub base_url: String,
    /// The `ilink_user_id` of whoever scanned the QR code.
    #[serde(default)]
    pub owners: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Credentials {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub telegram: Option<TelegramCred>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weixin: Option<WeixinCred>,
}

pub fn credentials_path() -> PathBuf {
    channels_dir().join("credentials.json")
}

pub fn load_credentials() -> Credentials {
    std::fs::read_to_string(credentials_path())
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

pub fn save_credentials(c: &Credentials) -> std::io::Result<()> {
    let text = serde_json::to_string_pretty(c)?;
    write_private(&credentials_path(), text.as_bytes())
}

/// Write atomically with 0600 permissions (tokens are secrets).
pub fn write_private(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)?;
    f.write_all(data)?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, path)
}

fn cursor_path(name: &str) -> PathBuf {
    channels_dir().join(format!("cursor-{}", super::safe_file_name(name)))
}

pub fn read_cursor(name: &str) -> Option<String> {
    std::fs::read_to_string(cursor_path(name))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn write_cursor(name: &str, value: &str) {
    if let Err(e) = write_private(&cursor_path(name), value.as_bytes()) {
        eprintln!("warning: cannot save cursor {name}: {e}");
    }
}

/// Held for the lifetime of the gateway process.
pub struct GatewayLock {
    _file: File,
}

impl GatewayLock {
    /// `Err` with a readable message when another gateway holds the lock.
    pub fn acquire() -> anyhow::Result<Self> {
        let path = channels_dir().join("gateway.lock");
        std::fs::create_dir_all(channels_dir())?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)?;
        // SAFETY: flock on a file descriptor we own; no memory is shared.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            anyhow::bail!(
                "another limbo gateway is already running ({} is locked)",
                path.display()
            );
        }
        let _ = file.set_len(0);
        let _ = writeln!(file, "{}", std::process::id());
        Ok(Self { _file: file })
    }
}
