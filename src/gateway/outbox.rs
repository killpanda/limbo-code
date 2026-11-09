//! One sender task per chat: delivers [`OutItem`]s in order, spaced by the
//! channel's `min_send_interval`, and decides what a failure means:
//!
//! - rate limited → wait `retry_after`, try again (up to 5 times);
//! - transient (network, 5xx) → back off 1/2/4 s, then give up on the item;
//! - reply context expired (WeChat) → **park** the item until the user
//!   writes again ([`OutItem::Resume`]), so a long task's answer is not lost;
//! - token expired → stop sending for this chat and log it once.
//!
//! Edits address *slots* (see `render.rs`); the platform handle of each slot
//! is known only here, after its first send.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

use super::render::OutItem;
use crate::channels::{Channel, MsgHandle, OutboundMessage, PeerKey, ReplyCtx, SendError};

const MAX_PARKED: usize = 50;

/// The latest reply context of a chat, shared by worker and sender.
pub type SharedCtx = Arc<Mutex<ReplyCtx>>;

pub fn spawn(
    channel: Arc<dyn Channel>,
    peer: PeerKey,
    ctx: SharedCtx,
) -> mpsc::UnboundedSender<OutItem> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(run(channel, peer, ctx, rx));
    tx
}

async fn run(
    channel: Arc<dyn Channel>,
    peer: PeerKey,
    ctx: SharedCtx,
    mut rx: mpsc::UnboundedReceiver<OutItem>,
) {
    let interval = channel.caps().min_send_interval;
    let mut handles: HashMap<u32, MsgHandle> = HashMap::new();
    let mut parked: Vec<OutItem> = Vec::new();
    let mut last: Option<Instant> = None;
    let mut dead = false;
    while let Some(item) = rx.recv().await {
        // Coalesce queued edits of the same slot: only the newest counts.
        let mut batch = vec![item];
        while let Ok(next) = rx.try_recv() {
            match (batch.last(), &next) {
                (Some(OutItem::Edit { slot: a, .. }), OutItem::Edit { slot: b, .. }) if a == b => {
                    *batch.last_mut().unwrap() = next;
                }
                _ => batch.push(next),
            }
        }
        for item in batch {
            if matches!(item, OutItem::Resume) {
                dead = false;
                let again = std::mem::take(&mut parked);
                for p in again {
                    deliver(
                        &*channel,
                        &peer,
                        &ctx,
                        p,
                        &mut handles,
                        &mut parked,
                        &mut last,
                        interval,
                        &mut dead,
                    )
                    .await;
                }
                continue;
            }
            if dead {
                continue;
            }
            deliver(
                &*channel,
                &peer,
                &ctx,
                item,
                &mut handles,
                &mut parked,
                &mut last,
                interval,
                &mut dead,
            )
            .await;
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn deliver(
    channel: &dyn Channel,
    peer: &PeerKey,
    ctx: &SharedCtx,
    item: OutItem,
    handles: &mut HashMap<u32, MsgHandle>,
    parked: &mut Vec<OutItem>,
    last: &mut Option<Instant>,
    interval: Duration,
    dead: &mut bool,
) {
    let (msg, slot) = match &item {
        OutItem::Send { msg, slot } => (msg.clone(), *slot),
        OutItem::Edit { slot, text, format } => match handles.get(slot) {
            Some(h) => (
                OutboundMessage::Edit {
                    handle: h.clone(),
                    text: text.clone(),
                    format: *format,
                },
                None,
            ),
            // Never sent (or the send failed): send it now.
            None => (
                OutboundMessage::Text {
                    text: text.clone(),
                    format: *format,
                },
                Some(*slot),
            ),
        },
        OutItem::Resume => return,
    };
    // Typing indicators are best effort: never retried, never parked.
    let typing = matches!(msg, OutboundMessage::Typing);
    let mut attempt = 0u32;
    loop {
        if let Some(t) = *last {
            let since = t.elapsed();
            if since < interval && !typing {
                tokio::time::sleep(interval - since).await;
            }
        }
        let c = ctx.lock().unwrap().clone();
        let res = channel.send(peer, &c, msg.clone()).await;
        if !typing {
            *last = Some(Instant::now());
        }
        match res {
            Ok(h) => {
                if let (Some(s), Some(h)) = (slot, h) {
                    handles.insert(s, h);
                }
                return;
            }
            Err(_) if typing => return,
            Err(SendError::RateLimited(d)) if attempt < 5 => {
                attempt += 1;
                tokio::time::sleep(d).await;
            }
            Err(SendError::Transient(e)) if attempt < 3 => {
                attempt += 1;
                eprintln!("gateway: send to {} failed ({e}), retrying", peer.key());
                tokio::time::sleep(Duration::from_secs(1 << (attempt - 1))).await;
            }
            Err(SendError::ContextExpired) => {
                if parked.len() < MAX_PARKED {
                    parked.push(item);
                }
                return;
            }
            Err(SendError::AuthExpired(e)) => {
                eprintln!(
                    "gateway: {} — {e}; messages to {} are dropped until you log in again",
                    peer.channel.as_str(),
                    peer.key()
                );
                *dead = true;
                return;
            }
            Err(e) => {
                eprintln!("gateway: send to {} failed: {e}", peer.key());
                return;
            }
        }
    }
}
