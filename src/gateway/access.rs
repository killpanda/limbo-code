//! Who may talk to the gateway: owners only. A chat front end is a remote
//! shell into this machine (limbo has no safety fences), so everyone not on
//! the list is dropped silently — no reply that would confirm a bot exists.
//!
//! Owners come from two places: `owners = [...]` in `[channels.<name>]` and
//! the account recorded at `limbo channels login` (the Telegram user who
//! sent the first message, the WeChat user who scanned the QR code).

use std::collections::{HashMap, HashSet};

use crate::channels::{ChannelKind, InboundMessage};

#[derive(Debug, Default, Clone)]
pub struct Access {
    owners: HashMap<ChannelKind, HashSet<String>>,
}

impl Access {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allow(&mut self, channel: ChannelKind, ids: impl IntoIterator<Item = String>) {
        self.owners.entry(channel).or_default().extend(
            ids.into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        );
    }

    pub fn owner_count(&self, channel: ChannelKind) -> usize {
        self.owners.get(&channel).map_or(0, HashSet::len)
    }

    /// The sender is an owner of this channel.
    pub fn admits(&self, msg: &InboundMessage) -> bool {
        self.owners
            .get(&msg.peer.channel)
            .is_some_and(|o| o.contains(&msg.sender))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::{PeerKey, ReplyCtx};

    fn msg(channel: ChannelKind, sender: &str) -> InboundMessage {
        InboundMessage {
            peer: PeerKey {
                channel,
                account: "b".into(),
                chat: sender.into(),
                thread: None,
            },
            sender: sender.into(),
            sender_name: String::new(),
            msg_id: "1".into(),
            text: "hi".into(),
            media: vec![],
            reply_ctx: ReplyCtx::default(),
            is_group: false,
            addressed: true,
        }
    }

    #[test]
    fn owners_are_per_channel() {
        let mut a = Access::new();
        a.allow(ChannelKind::Telegram, ["42".to_string()]);
        assert!(a.admits(&msg(ChannelKind::Telegram, "42")));
        assert!(!a.admits(&msg(ChannelKind::Telegram, "43")));
        assert!(!a.admits(&msg(ChannelKind::Weixin, "42")));
    }
}
