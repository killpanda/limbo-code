//! An in-memory channel for tests: records what the gateway sends.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::{
    Channel, ChannelCaps, ChannelKind, InboundMessage, MarkdownDialect, MediaRef, MsgHandle,
    OutboundMessage, PeerKey, ReplyCtx, SendError,
};

#[derive(Default, Clone)]
pub struct MockChannel {
    pub sent: Arc<Mutex<Vec<OutboundMessage>>>,
}

impl MockChannel {
    /// Plain texts sent so far (typing indicators left out).
    pub fn texts(&self) -> Vec<String> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .filter_map(|m| match m {
                OutboundMessage::Text { text, .. } | OutboundMessage::Edit { text, .. } => {
                    Some(text.clone())
                }
                OutboundMessage::File { path, .. } => Some(format!("file:{}", path.display())),
                _ => None,
            })
            .collect()
    }

    /// Wait until some sent text satisfies `f` (or time out).
    pub async fn wait_for(&self, f: impl Fn(&str) -> bool) -> bool {
        for _ in 0..200 {
            if self.texts().iter().any(|t| f(t)) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        false
    }
}

#[async_trait]
impl Channel for MockChannel {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Mock
    }
    fn account(&self) -> String {
        "mockbot".into()
    }
    fn caps(&self) -> ChannelCaps {
        ChannelCaps {
            max_text_chars: 1000,
            edit: false,
            markdown: MarkdownDialect::Plain,
            min_send_interval: Duration::ZERO,
            typing_ttl: Duration::from_secs(60),
        }
    }
    async fn run(
        &self,
        _inbox: mpsc::Sender<InboundMessage>,
        cancel: CancellationToken,
    ) -> anyhow::Result<()> {
        cancel.cancelled().await;
        Ok(())
    }
    async fn send(
        &self,
        _peer: &PeerKey,
        _ctx: &ReplyCtx,
        msg: OutboundMessage,
    ) -> Result<Option<MsgHandle>, SendError> {
        self.sent.lock().unwrap().push(msg);
        Ok(None)
    }
    async fn fetch_media(
        &self,
        media: &MediaRef,
        dir: &Path,
        _max: u64,
    ) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let p = dir.join(media.name.clone().unwrap_or_else(|| "f.txt".into()));
        std::fs::write(&p, "attached content")?;
        Ok(p)
    }
}
