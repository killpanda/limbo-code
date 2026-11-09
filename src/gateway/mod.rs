//! `limbo gateway`: talk to limbo from Telegram and personal WeChat.
//!
//! ```text
//!  channels::telegram ─┐                          ┌─ worker (chat A) ── Conversation ── Agent
//!  channels::weixin   ─┼─ inbox ── router ────────┼─ worker (chat B) ── …
//!                      │   dedupe · owners only   └─ …
//!                      └──────────── outbox (per chat) ◄── renderer ◄── AgentEvents
//! ```
//!
//! - adapters (`channels/`) long-poll their platform and push
//!   [`InboundMessage`]s into one inbox;
//! - the router drops duplicates, strangers and group chatter not aimed at
//!   the bot, then hands each message to its chat's worker (started on
//!   demand);
//! - each worker (`worker.rs`) owns one Conversation, renders its events
//!   (`render.rs`) and sends through its own outbox (`outbox.rs`).
//!
//! Also here: `limbo channels login|logout|status` and the background
//! service files for macOS (launchd) and Linux (systemd --user).

pub mod access;
pub mod outbox;
pub mod render;
pub mod send_file;
pub mod service;
pub mod sessions;
pub mod worker;

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::channels::store::{self, Credentials, TelegramCred};
use crate::channels::telegram::{self, Telegram};
use crate::channels::weixin::{self, Weixin};
use crate::channels::{Channel, ChannelKind, InboundMessage};
use crate::config::Config;
use access::Access;
use sessions::SessionMap;
use worker::WorkerEnv;

/// Remember this many message ids per channel for deduplication.
const DEDUP: usize = 2048;

fn expand_home(raw: &str) -> PathBuf {
    match raw.strip_prefix("~/") {
        Some(rest) => dirs::home_dir().unwrap_or_default().join(rest),
        None if raw == "~" => dirs::home_dir().unwrap_or_default(),
        None => PathBuf::from(raw),
    }
}

/// Recently seen message ids (bounded).
#[derive(Default)]
pub struct Dedup {
    seen: HashSet<String>,
    order: VecDeque<String>,
}

impl Dedup {
    /// `true` the first time `key` is seen.
    pub fn first(&mut self, key: String) -> bool {
        if !self.seen.insert(key.clone()) {
            return false;
        }
        self.order.push_back(key);
        if self.order.len() > DEDUP
            && let Some(old) = self.order.pop_front()
        {
            self.seen.remove(&old);
        }
        true
    }
}

/// What the router does with one message.
#[derive(Debug, PartialEq, Eq)]
pub enum Route {
    Deliver,
    Duplicate,
    Stranger,
    NotAddressed,
}

pub fn classify(msg: &InboundMessage, access: &Access, dedup: &mut Dedup) -> Route {
    if !dedup.first(format!("{}:{}", msg.peer.key(), msg.msg_id)) {
        return Route::Duplicate;
    }
    if !access.admits(msg) {
        return Route::Stranger;
    }
    if !msg.addressed {
        return Route::NotAddressed;
    }
    Route::Deliver
}

/// Build every configured, logged-in channel.
fn build_channels(
    config: &Config,
    creds: &Credentials,
    access: &mut Access,
) -> anyhow::Result<Vec<Arc<dyn Channel>>> {
    let mut out: Vec<Arc<dyn Channel>> = Vec::new();
    let tg = &config.channels.telegram;
    if tg.enabled
        && let Some(c) = &creds.telegram
    {
        let http = telegram::http_client(Some(&tg.proxy), Duration::from_secs(60))?;
        access.allow(
            ChannelKind::Telegram,
            tg.owners.iter().cloned().chain(c.owners.iter().cloned()),
        );
        out.push(Arc::new(Telegram::new(
            http,
            &tg.api_base,
            &c.token,
            &c.bot_id,
            &c.bot_username,
            tg.groups != "off",
        )));
    }
    let wx = &config.channels.weixin;
    if wx.enabled
        && let Some(c) = &creds.weixin
    {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .build()?;
        access.allow(
            ChannelKind::Weixin,
            wx.owners.iter().cloned().chain(c.owners.iter().cloned()),
        );
        out.push(Arc::new(Weixin::new(
            http,
            &c.base_url,
            &c.token,
            &c.bot_id,
            weixin::api::Identity::new(&wx.app_id, &wx.client_version),
        )));
    }
    Ok(out)
}

/// `limbo gateway`: run until Ctrl+C / SIGTERM.
pub async fn run(mut config: Config, session_dir: &Path) -> anyhow::Result<i32> {
    let _lock = store::GatewayLock::acquire()?;
    let g = config.gateway.clone();
    if !g.model.is_empty() {
        config.llm.model = g.model.clone();
    }
    let spec = crate::llm::catalog::resolve_model(&config.llm.model);
    if crate::llm::catalog::resolve_api_key(&spec, &config).is_none() {
        anyhow::bail!(
            "{}",
            crate::llm::catalog::missing_api_key_message(&spec, &config)
        );
    }
    let workdir = expand_home(&g.workdir);
    std::fs::create_dir_all(&workdir)
        .map_err(|e| anyhow::anyhow!("cannot create gateway workdir {}: {e}", workdir.display()))?;
    let workdir = std::fs::canonicalize(&workdir)?;
    let creds = store::load_credentials();
    let mut access = Access::new();
    let channels = build_channels(&config, &creds, &mut access)?;
    if channels.is_empty() {
        anyhow::bail!(
            "no channel is logged in: run `limbo channels login telegram` or `limbo channels login weixin` first"
        );
    }
    for ch in &channels {
        if access.owner_count(ch.kind()) == 0 {
            eprintln!(
                "warning: {} has no owners: nobody can talk to it (set owners in [channels.{}] or log in again)",
                ch.kind().as_str(),
                ch.kind().as_str()
            );
        }
    }
    let env = WorkerEnv {
        config: config.clone(),
        session_dir: session_dir.to_path_buf(),
        workdir: workdir.clone(),
        daily: g.posture != "coding",
        sessions: Arc::new(Mutex::new(SessionMap::load(
            &store::channels_dir().join("sessions.json"),
        ))),
        idle_unload: (g.idle_unload_secs > 0).then(|| Duration::from_secs(g.idle_unload_secs)),
        max_attachment: g.max_attachment_mb.max(1) * 1024 * 1024,
        show_tools: g.show_tools,
    };
    let cancel = CancellationToken::new();
    let (inbox_tx, mut inbox) = mpsc::channel::<InboundMessage>(256);
    let mut by_kind: HashMap<ChannelKind, Arc<dyn Channel>> = HashMap::new();
    for ch in &channels {
        by_kind.insert(ch.kind(), ch.clone());
        let (ch2, tx, c) = (ch.clone(), inbox_tx.clone(), cancel.clone());
        tokio::spawn(async move {
            if let Err(e) = ch2.run(tx, c).await {
                eprintln!("✗ {} stopped: {e:#}", ch2.kind().as_str());
            }
        });
    }
    drop(inbox_tx);
    println!(
        "limbo gateway · {} · workdir {} · {} · channels: {}",
        config.llm.model,
        workdir.display(),
        if env.daily { "daily" } else { "coding" },
        channels
            .iter()
            .map(|c| format!("{} ({})", c.kind().as_str(), c.account()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut workers: HashMap<String, mpsc::UnboundedSender<InboundMessage>> = HashMap::new();
    let mut dedup = Dedup::default();
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    loop {
        let msg = tokio::select! {
            _ = tokio::signal::ctrl_c() => break,
            _ = term.recv() => break,
            m = inbox.recv() => match m {
                Some(m) => m,
                None => {
                    eprintln!("all channels stopped");
                    break;
                }
            },
        };
        match classify(&msg, &access, &mut dedup) {
            Route::Deliver => {}
            Route::Stranger => {
                eprintln!(
                    "gateway: ignored a message from {} ({}) on {} — not an owner",
                    msg.sender,
                    msg.sender_name,
                    msg.peer.channel.as_str()
                );
                continue;
            }
            _ => continue,
        }
        let key = msg.peer.key();
        let Some(ch) = by_kind.get(&msg.peer.channel).cloned() else {
            continue;
        };
        let tx = workers.entry(key.clone()).or_insert_with(|| {
            worker::spawn(
                env.clone(),
                ch.clone(),
                msg.peer.clone(),
                cancel.child_token(),
            )
        });
        if let Err(mpsc::error::SendError(m)) = tx.send(msg) {
            // The worker ended (it never does on its own); start a new one.
            let tx = worker::spawn(env.clone(), ch, m.peer.clone(), cancel.child_token());
            let _ = tx.send(m);
            workers.insert(key, tx);
        }
    }
    println!("gateway stopping…");
    cancel.cancel();
    // Give workers a moment to checkpoint their sessions.
    tokio::time::sleep(Duration::from_millis(800)).await;
    Ok(0)
}

// -- `limbo channels …` --------------------------------------------------------

pub async fn login(config: &Config, channel: &str, token: Option<String>) -> anyhow::Result<i32> {
    match ChannelKind::parse(channel) {
        Some(ChannelKind::Telegram) => login_telegram(config, token).await,
        Some(ChannelKind::Weixin) => {
            let wx = &config.channels.weixin;
            let http = reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .build()?;
            let cred = weixin::login::login(
                http,
                weixin::api::Identity::new(&wx.app_id, &wx.client_version),
            )
            .await?;
            let mut creds = store::load_credentials();
            println!(
                "✓ 微信已连接（bot {}）；主人：{}",
                cred.bot_id,
                cred.owners
                    .first()
                    .map(String::as_str)
                    .unwrap_or("（未返回，请在 [channels.weixin] owners 里配置）")
            );
            creds.weixin = Some(cred);
            store::save_credentials(&creds)?;
            println!("运行 `limbo gateway` 开始接收消息。");
            Ok(0)
        }
        _ => anyhow::bail!("unknown channel {channel:?} (telegram | weixin)"),
    }
}

async fn login_telegram(config: &Config, token: Option<String>) -> anyhow::Result<i32> {
    let tg = &config.channels.telegram;
    let token = match token {
        Some(t) => t,
        None => {
            println!(
                "在 Telegram 里找 @BotFather，/newbot 创建机器人，把它给你的 token 粘贴到这里："
            );
            let mut s = String::new();
            std::io::stdin().read_line(&mut s)?;
            s
        }
    };
    let token = token.trim().to_string();
    if token.is_empty() {
        anyhow::bail!("empty token");
    }
    let http = telegram::http_client(Some(&tg.proxy), Duration::from_secs(60))?;
    let (bot_id, username) = telegram::get_me(&http, &tg.api_base, &token)
        .await
        .map_err(|e| anyhow::anyhow!("token check failed: {e}"))?;
    println!("✓ bot @{username}（id {bot_id}）");
    let mut cred = TelegramCred {
        token: token.clone(),
        bot_id: bot_id.clone(),
        bot_username: username.clone(),
        owners: store::load_credentials()
            .telegram
            .filter(|c| c.bot_id == bot_id)
            .map(|c| c.owners)
            .unwrap_or_default(),
    };
    if cred.owners.is_empty() && tg.owners.is_empty() {
        println!(
            "现在用你自己的 Telegram 账号给 @{username} 发一条消息（例如 /start），把你登记为主人…"
        );
        let bot = Telegram::new(http, &tg.api_base, &token, &bot_id, &username, false);
        let (tx, mut rx) = mpsc::channel(8);
        let cancel = CancellationToken::new();
        let c2 = cancel.clone();
        let poll = tokio::spawn(async move { bot.run(tx, c2).await });
        let first = tokio::time::timeout(Duration::from_secs(300), async {
            loop {
                match rx.recv().await {
                    Some(m) if !m.is_group => return Some(m),
                    Some(_) => continue,
                    None => return None,
                }
            }
        })
        .await;
        cancel.cancel();
        let _ = poll.await;
        match first {
            Ok(Some(m)) => {
                println!("✓ 主人：{} ({})", m.sender, m.sender_name);
                cred.owners.push(m.sender);
            }
            _ => println!(
                "没有收到消息；之后可以在 config.toml 的 [channels.telegram] owners 里填你的 user id。"
            ),
        }
    }
    let mut creds = store::load_credentials();
    creds.telegram = Some(cred);
    store::save_credentials(&creds)?;
    println!(
        "已保存到 {}；运行 `limbo gateway` 开始接收消息。",
        store::credentials_path().display()
    );
    Ok(0)
}

pub fn logout(channel: &str) -> anyhow::Result<i32> {
    let mut creds = store::load_credentials();
    match ChannelKind::parse(channel) {
        Some(ChannelKind::Telegram) => creds.telegram = None,
        Some(ChannelKind::Weixin) => creds.weixin = None,
        _ => anyhow::bail!("unknown channel {channel:?} (telegram | weixin)"),
    }
    store::save_credentials(&creds)?;
    println!("已删除 {channel} 的登录信息");
    Ok(0)
}

pub fn status(config: &Config) -> i32 {
    let creds = store::load_credentials();
    let line = |name: &str,
                enabled: bool,
                logged: Option<(String, Vec<String>)>,
                cfg_owners: &[String]| {
        let state = match (&logged, enabled) {
            (None, _) => "未登录".to_string(),
            (Some(_), false) => "已登录（config 中已停用）".to_string(),
            (Some((bot, _)), true) => format!("已登录 · bot {bot}"),
        };
        let mut owners: Vec<String> = cfg_owners.to_vec();
        if let Some((_, o)) = &logged {
            owners.extend(o.iter().cloned());
        }
        println!(
            "{name:9} {state} · 主人 {}",
            if owners.is_empty() {
                "（无）".into()
            } else {
                owners.join(", ")
            }
        );
    };
    line(
        "telegram",
        config.channels.telegram.enabled,
        creds
            .telegram
            .map(|c| (format!("@{} ({})", c.bot_username, c.bot_id), c.owners)),
        &config.channels.telegram.owners,
    );
    line(
        "weixin",
        config.channels.weixin.enabled,
        creds.weixin.map(|c| (c.bot_id, c.owners)),
        &config.channels.weixin.owners,
    );
    println!(
        "会话映射：{}",
        store::channels_dir().join("sessions.json").display()
    );
    println!("后台服务：{}", service::describe());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::{PeerKey, ReplyCtx};

    fn msg(sender: &str, id: &str, addressed: bool) -> InboundMessage {
        InboundMessage {
            peer: PeerKey {
                channel: ChannelKind::Telegram,
                account: "b".into(),
                chat: "c".into(),
                thread: None,
            },
            sender: sender.into(),
            sender_name: String::new(),
            msg_id: id.into(),
            text: "hi".into(),
            media: vec![],
            reply_ctx: ReplyCtx::default(),
            is_group: !addressed,
            addressed,
        }
    }

    #[test]
    fn router_rules() {
        let mut a = Access::new();
        a.allow(ChannelKind::Telegram, ["42".to_string()]);
        let mut d = Dedup::default();
        assert_eq!(classify(&msg("42", "1", true), &a, &mut d), Route::Deliver);
        assert_eq!(
            classify(&msg("42", "1", true), &a, &mut d),
            Route::Duplicate
        );
        assert_eq!(classify(&msg("7", "2", true), &a, &mut d), Route::Stranger);
        assert_eq!(
            classify(&msg("42", "3", false), &a, &mut d),
            Route::NotAddressed
        );
    }

    #[test]
    fn dedup_is_bounded() {
        let mut d = Dedup::default();
        for i in 0..(DEDUP + 10) {
            assert!(d.first(i.to_string()));
        }
        assert!(d.seen.len() <= DEDUP);
        assert!(d.first("0".into()), "oldest ids are forgotten");
    }
}
