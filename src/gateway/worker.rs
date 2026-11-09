//! One worker per chat: a tokio task that owns that chat's
//! [`Conversation`] and turns its traffic into chat messages.
//!
//! The loop mirrors the headless REPL's (`headless::run_repl`): inbound
//! messages, agent events of the running turn, the turn finishing,
//! background events of this session (subagents) and a ticker for streaming
//! previews, typing indicators and unloading an idle agent.
//!
//! - idle + message → a new turn (`pump::run_task` via the Conversation);
//! - busy + message → *steer* into the running turn (`/queue` waits);
//! - `/stop` interrupts; `/sessions` and `/resume <id>` switch between this
//!   chat's sessions; everything else is the usual slash commands.
//!
//! 🦀 Rust notes
//! - `conv: Option<Conversation<..>>`: the agent is loaded lazily on the
//!   first message and dropped again after `idle_unload_secs`, so a
//!   gateway with many quiet chats holds no agents in memory.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use super::outbox::{self, SharedCtx};
use super::render::{OutItem, Renderer};
use super::send_file::SendFileTool;
use super::sessions::SessionMap;
use crate::agent::{Agent, AgentEvent};
use crate::background::{self, BackgroundEvent};
use crate::channels::{Channel, ChannelKind, InboundMessage, MediaKind, OutboundMessage, PeerKey};
use crate::config::Config;
use crate::conversation::{Conversation, Sink};
use crate::integrations::Reporters;
use crate::models::Attachment;

/// Everything a worker needs besides its chat.
#[derive(Clone)]
pub struct WorkerEnv {
    pub config: Config,
    pub session_dir: PathBuf,
    pub workdir: PathBuf,
    pub daily: bool,
    pub sessions: Arc<Mutex<SessionMap>>,
    pub idle_unload: Option<Duration>,
    pub max_attachment: u64,
    pub show_tools: bool,
}

/// The Conversation's output in a chat: lines become plain messages.
struct ChatSink {
    out: mpsc::UnboundedSender<OutItem>,
    btw: String,
    channel: ChannelKind,
    max_attachment: u64,
}

impl ChatSink {
    fn install_tools(&self, agent: &mut Agent) {
        agent.registry.add(Arc::new(SendFileTool {
            out: self.out.clone(),
            channel: match self.channel {
                ChannelKind::Telegram => "Telegram",
                ChannelKind::Weixin => "WeChat",
                #[cfg(test)]
                ChannelKind::Mock => "mock",
            },
            max_bytes: self.max_attachment.max(50 * 1024 * 1024),
        }));
        agent.init_system_message();
    }
}

impl Sink for ChatSink {
    fn line(&mut self, text: &str) {
        let _ = self.out.send(OutItem::text(text));
    }
    fn btw_chunk(&mut self, text: &str) {
        self.btw.push_str(text);
    }
    fn btw_end(&mut self) {
        let t = std::mem::take(&mut self.btw);
        if !t.trim().is_empty() {
            let _ = self.out.send(OutItem::Send {
                msg: OutboundMessage::Text {
                    text: t,
                    format: crate::channels::TextFormat::Markdown,
                },
                slot: None,
            });
        }
    }
    fn agent_replaced(&mut self, agent: &mut Agent) {
        self.install_tools(agent);
    }
    fn allows_quit(&self) -> bool {
        false
    }
}

pub const CHAT_HELP: &str = "聊天里额外可用：/stop 打断当前任务 · /sessions 本聊天的会话 · /resume <id> 切回旧会话 · /status 当前状态";

/// Start the worker of `peer`; returns its mailbox.
pub fn spawn(
    env: WorkerEnv,
    channel: Arc<dyn Channel>,
    peer: PeerKey,
    cancel: CancellationToken,
) -> mpsc::UnboundedSender<InboundMessage> {
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        Worker::new(env, channel, peer).run(rx, cancel).await;
    });
    tx
}

struct Worker {
    env: WorkerEnv,
    channel: Arc<dyn Channel>,
    peer: PeerKey,
    ctx: SharedCtx,
    out: mpsc::UnboundedSender<OutItem>,
    conv: Option<Conversation<ChatSink>>,
    ev_rx: Option<mpsc::UnboundedReceiver<AgentEvent>>,
    bg_tx: mpsc::UnboundedSender<BackgroundEvent>,
    bg_rx: mpsc::UnboundedReceiver<BackgroundEvent>,
    renderer: Renderer,
    routed: String,
    last_activity: Instant,
    last_typing: Option<Instant>,
}

impl Worker {
    fn new(env: WorkerEnv, channel: Arc<dyn Channel>, peer: PeerKey) -> Self {
        let ctx: SharedCtx = Arc::new(Mutex::new(Default::default()));
        let out = outbox::spawn(channel.clone(), peer.clone(), ctx.clone());
        let (bg_tx, bg_rx) = mpsc::unbounded_channel();
        let renderer = Renderer::new(channel.caps(), env.show_tools);
        Self {
            env,
            channel,
            peer,
            ctx,
            out,
            conv: None,
            ev_rx: None,
            bg_tx,
            bg_rx,
            renderer,
            routed: String::new(),
            last_activity: Instant::now(),
            last_typing: None,
        }
    }

    fn say(&self, text: impl Into<String>) {
        let _ = self.out.send(OutItem::text(text));
    }

    fn busy(&self) -> bool {
        self.conv.as_ref().is_some_and(|c| c.busy())
    }

    /// Load (resume or create) this chat's agent.
    fn load(&mut self, session: Option<&str>) -> anyhow::Result<()> {
        let current = match session {
            Some(s) => Some(s.to_string()),
            None => self.env.sessions.lock().unwrap().current(&self.peer),
        };
        let resumed = current
            .as_deref()
            .and_then(|id| crate::session::find_session(&self.env.session_dir, id).ok())
            .and_then(|p| Agent::resume(self.env.config.clone(), &p, &self.env.session_dir).ok());
        let mut agent = match resumed {
            Some(a) => a,
            None => {
                if session.is_some() {
                    anyhow::bail!("找不到会话 {}", session.unwrap_or(""));
                }
                Agent::create(
                    self.env.config.clone(),
                    &self.env.workdir,
                    &self.env.session_dir,
                    self.env.daily,
                )?
            }
        };
        let sink = ChatSink {
            out: self.out.clone(),
            btw: String::new(),
            channel: self.peer.channel,
            max_attachment: self.env.max_attachment,
        };
        sink.install_tools(&mut agent);
        let (tx, rx) = mpsc::unbounded_channel();
        // Chat sessions report nothing to terminal integrations.
        let reporters = Arc::new(Reporters::default());
        reporters.release();
        self.conv = Some(Conversation::new(agent, sink, tx, reporters));
        self.ev_rx = Some(rx);
        self.sync_session();
        Ok(())
    }

    /// Keep the session map, the origin tag and the background route in
    /// step with the agent's current session.
    fn sync_session(&mut self) {
        let Some(agent) = self.conv.as_mut().and_then(|c| c.agent.as_mut()) else {
            return;
        };
        let id = agent.session_id().to_string();
        if id == self.routed {
            return;
        }
        if !self.routed.is_empty() {
            background::unroute(&self.routed);
        }
        background::route(&id, self.bg_tx.clone());
        self.routed = id.clone();
        agent
            .meta
            .extra
            .insert("origin".into(), serde_json::json!(self.peer.key()));
        agent.persist_meta();
        self.env.sessions.lock().unwrap().set(&self.peer, &id);
    }

    async fn unload(&mut self) {
        if let Some(mut c) = self.conv.take() {
            c.shutdown().await;
        }
        self.ev_rx = None;
        if !self.routed.is_empty() {
            background::unroute(&self.routed);
            self.routed.clear();
        }
    }

    async fn run(
        mut self,
        mut inbox: mpsc::UnboundedReceiver<InboundMessage>,
        cancel: CancellationToken,
    ) {
        let mut ticker = tokio::time::interval(self.renderer.tick_interval());
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            let busy = self.busy();
            let has_events = self.ev_rx.is_some();
            // Disjoint borrows for the branch futures; the handlers below run
            // after those futures are dropped and may use `self` freely.
            let Worker {
                ev_rx, conv, bg_rx, ..
            } = &mut self;
            tokio::select! {
                _ = cancel.cancelled() => break,
                m = inbox.recv() => match m {
                    Some(m) => self.on_message(m).await,
                    None => break,
                },
                Some(ev) = async { ev_rx.as_mut().unwrap().recv().await }, if has_events => {
                    self.on_event(ev);
                }
                Some(done) = async { conv.as_mut().unwrap().join().await }, if busy => {
                    self.turn_done(done);
                }
                Some(ev) = bg_rx.recv() => {
                    if let Some(c) = self.conv.as_mut() {
                        for l in c.background_lines(&ev) {
                            c.sink.line(&l);
                        }
                        c.after_background(&ev);
                    }
                }
                _ = ticker.tick() => self.on_tick().await,
            }
        }
        self.unload().await;
    }

    fn turn_done(&mut self, done: Result<Agent, tokio::task::JoinError>) {
        // Events emitted just before the task returned.
        while let Some(ev) = self.ev_rx.as_mut().and_then(|r| r.try_recv().ok()) {
            self.on_event(ev);
        }
        match done {
            Ok(agent) => {
                if let Some(c) = self.conv.as_mut() {
                    c.job_done(agent);
                }
                self.sync_session();
            }
            Err(e) => {
                self.say(format!("✗ 内部错误：{e}"));
                self.conv = None;
                self.ev_rx = None;
            }
        }
        self.last_activity = Instant::now();
    }

    fn on_event(&mut self, ev: AgentEvent) {
        if let Some(c) = self.conv.as_mut() {
            c.observe(&ev);
        }
        for item in self.renderer.on_event(&ev, Instant::now()) {
            let _ = self.out.send(item);
        }
    }

    async fn on_tick(&mut self) {
        let now = Instant::now();
        for item in self.renderer.tick(now) {
            let _ = self.out.send(item);
        }
        if self.busy() {
            let ttl = self.channel.caps().typing_ttl;
            if self
                .last_typing
                .is_none_or(|t| now.duration_since(t) >= ttl)
            {
                self.last_typing = Some(now);
                let _ = self.out.send(OutItem::Send {
                    msg: OutboundMessage::Typing,
                    slot: None,
                });
            }
            self.last_activity = now;
            return;
        }
        self.last_typing = None;
        if let Some(limit) = self.env.idle_unload
            && self
                .conv
                .as_ref()
                .is_some_and(|c| c.proposals.is_none() && !c.sup().is_some_and(|s| s.working()))
            && now.duration_since(self.last_activity) >= limit
        {
            self.unload().await;
        }
    }

    async fn on_message(&mut self, m: InboundMessage) {
        self.last_activity = Instant::now();
        if m.reply_ctx.0.is_some() {
            *self.ctx.lock().unwrap() = m.reply_ctx.clone();
        }
        let _ = self.out.send(OutItem::Resume);
        let text = m.text.trim().to_string();
        let (cmd, arg) = match text.strip_prefix('/') {
            Some(rest) => rest
                .split_once(char::is_whitespace)
                .map_or((rest, ""), |(c, a)| (c, a.trim())),
            None => ("", ""),
        };
        match cmd.to_lowercase().as_str() {
            "stop" => {
                match self.conv.as_mut() {
                    Some(c) if c.busy() => c.interrupt(),
                    _ => self.say("当前没有进行中的任务"),
                }
                return;
            }
            "start" if self.conv.is_none() && arg.is_empty() => {
                self.say(format!(
                    "limbo 已连接。直接发消息开始；/help 查看命令。\n{CHAT_HELP}"
                ));
                return;
            }
            "sessions" => {
                let known = self.env.sessions.lock().unwrap().known(&self.peer);
                if known.is_empty() {
                    self.say("这个聊天还没有会话");
                } else {
                    let mut lines = vec!["本聊天的会话（第一个是当前会话）：".to_string()];
                    lines.extend(known.iter().map(|s| format!("• {s}")));
                    lines.push("/resume <id 前缀> 切换".into());
                    self.say(lines.join("\n"));
                }
                return;
            }
            "resume" => {
                return self.resume(arg).await;
            }
            "status" => {
                let s = match self.conv.as_ref().and_then(|c| c.agent.as_ref()) {
                    Some(a) => format!(
                        "会话 {} · 模型 {} · {} · 累计 {} tokens",
                        a.session_id(),
                        a.client.model_id(),
                        if a.daily_mode() { "daily" } else { "coding" },
                        a.session_total_tokens()
                    ),
                    None if self.busy() => "任务进行中".into(),
                    None => "空闲（下一条消息时加载会话）".into(),
                };
                self.say(s);
                return;
            }
            _ => {}
        }
        if self.conv.is_none()
            && let Err(e) = self.load(None)
        {
            self.say(format!("✗ 无法启动会话：{e}"));
            return;
        }
        let attachments = self.download(&m).await;
        if text.is_empty() && attachments.is_empty() {
            return;
        }
        let Some(c) = self.conv.as_mut() else { return };
        c.submit(text.clone(), attachments).await;
        if cmd.eq_ignore_ascii_case("help") {
            self.say(CHAT_HELP);
        }
        self.sync_session();
    }

    async fn resume(&mut self, arg: &str) {
        if self.busy() {
            return self.say("当前任务进行中；先 /stop 或等待完成");
        }
        let known = self.env.sessions.lock().unwrap().known(&self.peer);
        let matches: Vec<&String> = known
            .iter()
            .filter(|s| !arg.is_empty() && s.starts_with(arg))
            .collect();
        let id = match matches.as_slice() {
            [one] => (*one).clone(),
            [] => return self.say("用法：/resume <id 前缀>（/sessions 查看本聊天的会话）"),
            _ => return self.say("前缀对应多个会话，请写长一点"),
        };
        self.unload().await;
        match self.load(Some(&id)) {
            Ok(()) => self.say(format!("已切换到会话 {id}")),
            Err(e) => {
                self.say(format!("✗ {e}"));
                let _ = self.load(None);
            }
        }
    }

    /// Download the message's media next to the session file.
    async fn download(&self, m: &InboundMessage) -> Vec<Attachment> {
        if m.media.is_empty() {
            return vec![];
        }
        let dir = self
            .env
            .session_dir
            .join("media")
            .join(crate::channels::safe_file_name(&self.routed))
            .join(crate::channels::safe_file_name(&m.msg_id));
        let mut out = Vec::new();
        for media in &m.media {
            match self
                .channel
                .fetch_media(media, &dir, self.env.max_attachment)
                .await
            {
                Ok(p) => out.push(crate::attachments::attachment_for(&p)),
                Err(e) => {
                    let what = match media.kind {
                        MediaKind::Image => "图片",
                        MediaKind::File => "文件",
                        MediaKind::Voice => "语音",
                        MediaKind::Video => "视频",
                    };
                    self.say(format!("✗ {what}下载失败：{e}"))
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tests::{fake_server, sse, test_config};
    use crate::channels::ReplyCtx;
    use crate::channels::mock::MockChannel;
    use serde_json::json;

    fn msg(peer: &PeerKey, id: &str, text: &str) -> InboundMessage {
        InboundMessage {
            peer: peer.clone(),
            sender: "42".into(),
            sender_name: "owner".into(),
            msg_id: id.into(),
            text: text.into(),
            media: vec![],
            reply_ctx: ReplyCtx(None),
            is_group: false,
            addressed: true,
        }
    }

    /// A chat message runs a turn end to end: the reply reaches the chat,
    /// the chat → session mapping is recorded, and `/status` answers.
    #[tokio::test]
    async fn message_runs_a_turn_and_records_the_session() {
        let dir = tempfile::tempdir().unwrap();
        let reply = sse(&[
            json!({"choices":[{"delta":{"content":"你好，我在。"},"finish_reason":"stop"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":10,"total_tokens":12}}),
        ]);
        let (base, seen) = fake_server(vec![reply]).await;
        let workdir = dir.path().join("work");
        std::fs::create_dir_all(&workdir).unwrap();
        let sessions = Arc::new(Mutex::new(SessionMap::load(
            &dir.path().join("sessions.json"),
        )));
        let env = WorkerEnv {
            config: test_config(&base),
            session_dir: dir.path().join("sessions"),
            workdir,
            daily: true,
            sessions: sessions.clone(),
            idle_unload: None,
            max_attachment: 1024 * 1024,
            show_tools: true,
        };
        let mock = MockChannel::default();
        let peer = PeerKey {
            channel: ChannelKind::Mock,
            account: "mockbot".into(),
            chat: "42".into(),
            thread: None,
        };
        let cancel = CancellationToken::new();
        let tx = spawn(env, Arc::new(mock.clone()), peer.clone(), cancel.clone());
        tx.send(msg(&peer, "1", "在吗")).unwrap();
        assert!(
            mock.wait_for(|t| t.contains("你好，我在。")).await,
            "{:?}",
            mock.texts()
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert!(sessions.lock().unwrap().current(&peer).is_some());

        tx.send(msg(&peer, "2", "/status")).unwrap();
        assert!(
            mock.wait_for(|t| t.contains("模型 test-model")).await,
            "{:?}",
            mock.texts()
        );
        cancel.cancel();
    }
}
