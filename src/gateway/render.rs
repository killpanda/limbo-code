//! Agent events → chat messages. A small state machine, free of I/O so it
//! can be tested with plain event lists:
//!
//! - **answer text** accumulates per segment (a segment ends at a tool call,
//!   a notice or the turn end). Channels that can edit (Telegram) see it
//!   stream: one message is sent and then edited at most once per
//!   `min_send_interval` as plain text, and replaced by the formatted
//!   markdown when the segment is complete. Other channels (WeChat) get the
//!   finished segment only.
//! - **tool calls** become one progress message, edited as calls come in
//!   (editing channels, `[gateway] show_tools`); elsewhere a typing
//!   indicator is all the user sees while tools run.
//! - `ask_user` becomes a question with options (buttons on Telegram).
//!
//! Output goes to the outbox as [`OutItem`]s; editable messages are named
//! by *slots* (small numbers) because the platform's message id is only
//! known once the outbox has sent them.

use std::time::{Duration, Instant};

use crate::agent::AgentEvent;
use crate::channels::{ChannelCaps, MarkdownDialect, OutboundMessage, TextFormat, markdown};

/// Something for the outbox to deliver.
#[derive(Debug, Clone)]
pub enum OutItem {
    /// Send; when `slot` is set, remember the message for later edits.
    Send {
        msg: OutboundMessage,
        slot: Option<u32>,
    },
    /// Replace the text of the message in `slot` (sent fresh if unknown).
    Edit {
        slot: u32,
        text: String,
        format: TextFormat,
    },
    /// A new inbound message arrived: retry parked sends.
    Resume,
}

impl OutItem {
    pub fn text(text: impl Into<String>) -> Self {
        OutItem::Send {
            msg: OutboundMessage::Text {
                text: text.into(),
                format: TextFormat::Plain,
            },
            slot: None,
        }
    }
}

const MAX_TOOL_LINES: usize = 8;
const CURSOR: &str = " ▍";

pub struct Renderer {
    caps: ChannelCaps,
    show_tools: bool,
    text: String,
    /// Streaming preview of the current segment: (slot, text last sent).
    live: Option<(u32, String)>,
    tools: Vec<String>,
    tool_count: usize,
    tool_slot: Option<u32>,
    tools_dirty: bool,
    next_slot: u32,
    last_flush: Option<Instant>,
}

impl Renderer {
    pub fn new(caps: ChannelCaps, show_tools: bool) -> Self {
        Self {
            caps,
            show_tools,
            text: String::new(),
            live: None,
            tools: vec![],
            tool_count: 0,
            tool_slot: None,
            tools_dirty: false,
            next_slot: 1,
            last_flush: None,
        }
    }

    fn slot(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    /// Room for one chunk of markdown source: Telegram's limit counts the
    /// rendered text, and HTML tags / entities add a little.
    fn chunk_budget(&self) -> usize {
        match self.caps.markdown {
            MarkdownDialect::TelegramHtml => self.caps.max_text_chars * 3 / 4,
            MarkdownDialect::Plain => self.caps.max_text_chars,
        }
    }

    pub fn on_event(&mut self, ev: &AgentEvent, now: Instant) -> Vec<OutItem> {
        let mut out = Vec::new();
        if let Some(line) = crate::commands::goal_event_line(ev) {
            self.finish_segment(&mut out);
            out.push(OutItem::text(line));
            return out;
        }
        match ev {
            AgentEvent::TurnStart => {
                self.tools.clear();
                self.tool_count = 0;
                self.tool_slot = None;
                self.tools_dirty = false;
            }
            AgentEvent::Text(t) => {
                self.text.push_str(t);
                // A very long answer: hand complete chunks over right away.
                if self.text.chars().count() > self.chunk_budget() {
                    let mut chunks = markdown::split(&self.text, self.chunk_budget());
                    let last = chunks.pop().unwrap_or_default();
                    for c in chunks {
                        self.emit_final(&mut out, c);
                    }
                    self.text = last;
                }
            }
            AgentEvent::Retry { .. } => {
                // The streamed half is void; the live message is overwritten.
                self.text.clear();
            }
            AgentEvent::ToolCall {
                name, arguments, ..
            } => {
                self.finish_segment(&mut out);
                if name == "ask_user" {
                    let question = arguments
                        .get("question")
                        .and_then(|q| q.as_str())
                        .unwrap_or("")
                        .to_string();
                    let options = arguments
                        .get("options")
                        .and_then(|o| o.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    out.push(OutItem::Send {
                        msg: OutboundMessage::Choices { question, options },
                        slot: None,
                    });
                } else if self.caps.edit && self.show_tools {
                    self.tool_count += 1;
                    let desc = if name == "run_code" {
                        arguments
                            .get("description")
                            .and_then(|d| d.as_str())
                            .map(|d| format!("run_code · {}", d.lines().next().unwrap_or("")))
                            .unwrap_or_else(|| crate::subagents::describe_call(name, arguments))
                    } else {
                        crate::subagents::describe_call(name, arguments)
                    };
                    self.tools
                        .push(format!("⚙ {}", crate::llm::truncate_chars(&desc, 80)));
                    if self.tools.len() > MAX_TOOL_LINES {
                        self.tools.remove(0);
                    }
                    self.tools_dirty = true;
                }
            }
            AgentEvent::ToolResult { success, .. } => {
                if !success
                    && let Some(l) = self.tools.last_mut()
                    && l.starts_with('⚙')
                {
                    *l = l.replacen('⚙', "✗", 1);
                    self.tools_dirty = true;
                }
            }
            AgentEvent::Info(t) => {
                self.finish_segment(&mut out);
                out.push(OutItem::text(t.clone()));
            }
            AgentEvent::Warning(t) => {
                self.finish_segment(&mut out);
                out.push(OutItem::text(format!("⚠ {t}")));
            }
            AgentEvent::Error(t) => {
                self.finish_segment(&mut out);
                out.push(OutItem::text(format!("✗ {t}")));
            }
            AgentEvent::Interrupted => {
                self.finish_segment(&mut out);
                out.push(OutItem::text("⏹ 已打断"));
            }
            AgentEvent::Compaction {
                compacted: true,
                tokens_before,
                tokens_after,
                ..
            } => out.push(OutItem::text(format!(
                "已压缩上下文：约 {tokens_before} → {tokens_after} tokens"
            ))),
            AgentEvent::TurnEnd => {
                self.finish_segment(&mut out);
                self.flush_tools(&mut out, true);
            }
            _ => {}
        }
        if self.tools_dirty && self.due(now) {
            self.flush_tools(&mut out, false);
            self.last_flush = Some(now);
        }
        out
    }

    fn due(&self, now: Instant) -> bool {
        self.last_flush
            .is_none_or(|t| now.duration_since(t) >= self.caps.min_send_interval)
    }

    /// Periodic work: stream the answer preview and the tool list.
    pub fn tick(&mut self, now: Instant) -> Vec<OutItem> {
        let mut out = Vec::new();
        if !self.caps.edit || !self.due(now) {
            return out;
        }
        let mut flushed = false;
        if self.tools_dirty {
            self.flush_tools(&mut out, false);
            flushed = true;
        }
        let preview = self.text.trim_end();
        if !preview.is_empty() {
            let shown = format!("{preview}{CURSOR}");
            match &self.live {
                Some((_, last)) if *last == shown => {}
                Some((slot, _)) => {
                    let slot = *slot;
                    out.push(OutItem::Edit {
                        slot,
                        text: shown.clone(),
                        format: TextFormat::Plain,
                    });
                    self.live = Some((slot, shown));
                    flushed = true;
                }
                None => {
                    let slot = self.slot();
                    out.push(OutItem::Send {
                        msg: OutboundMessage::Text {
                            text: shown.clone(),
                            format: TextFormat::Plain,
                        },
                        slot: Some(slot),
                    });
                    self.live = Some((slot, shown));
                    flushed = true;
                }
            }
        }
        if flushed {
            self.last_flush = Some(now);
        }
        out
    }

    fn flush_tools(&mut self, out: &mut Vec<OutItem>, done: bool) {
        if !self.tools_dirty && !(done && self.tool_slot.is_some()) {
            return;
        }
        self.tools_dirty = false;
        if self.tools.is_empty() {
            return;
        }
        let mut text = self.tools.join("\n");
        if self.tool_count > self.tools.len() {
            text = format!("…（共 {} 次工具调用）\n{text}", self.tool_count);
        }
        if done {
            text.push_str(&format!("\n✓ 本轮调用工具 {} 次", self.tool_count));
        }
        match self.tool_slot {
            Some(slot) => out.push(OutItem::Edit {
                slot,
                text,
                format: TextFormat::Plain,
            }),
            None => {
                let slot = self.slot();
                self.tool_slot = Some(slot);
                out.push(OutItem::Send {
                    msg: OutboundMessage::Text {
                        text,
                        format: TextFormat::Plain,
                    },
                    slot: Some(slot),
                });
            }
        }
        if done {
            self.tool_slot = None;
            self.tools.clear();
        }
    }

    /// The current segment is complete: send it formatted.
    fn finish_segment(&mut self, out: &mut Vec<OutItem>) {
        let text = std::mem::take(&mut self.text);
        if text.trim().is_empty() {
            if let Some((slot, _)) = self.live.take() {
                out.push(OutItem::Edit {
                    slot,
                    text: "…".into(),
                    format: TextFormat::Plain,
                });
            }
            return;
        }
        for chunk in markdown::split(text.trim(), self.chunk_budget()) {
            self.emit_final(out, chunk);
        }
    }

    /// One finished chunk: replaces the live preview if there is one.
    fn emit_final(&mut self, out: &mut Vec<OutItem>, chunk: String) {
        match self.live.take() {
            Some((slot, _)) => out.push(OutItem::Edit {
                slot,
                text: chunk,
                format: TextFormat::Markdown,
            }),
            None => out.push(OutItem::Send {
                msg: OutboundMessage::Text {
                    text: chunk,
                    format: TextFormat::Markdown,
                },
                slot: None,
            }),
        }
    }

    /// How often the worker should call [`Renderer::tick`].
    pub fn tick_interval(&self) -> Duration {
        self.caps
            .min_send_interval
            .clamp(Duration::from_millis(50), Duration::from_millis(500))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caps(edit: bool) -> ChannelCaps {
        ChannelCaps {
            max_text_chars: 400,
            edit,
            markdown: MarkdownDialect::Plain,
            min_send_interval: Duration::from_secs(1),
            typing_ttl: Duration::from_secs(5),
        }
    }

    fn texts(items: &[OutItem]) -> Vec<String> {
        items
            .iter()
            .map(|i| match i {
                OutItem::Send {
                    msg: OutboundMessage::Text { text, .. },
                    slot,
                } => format!(
                    "send{}:{text}",
                    slot.map(|s| format!("#{s}")).unwrap_or_default()
                ),
                OutItem::Edit { slot, text, .. } => format!("edit#{slot}:{text}"),
                OutItem::Send {
                    msg: OutboundMessage::Choices { question, options },
                    ..
                } => format!("ask:{question}:{}", options.join("|")),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn non_editing_channel_gets_finished_segments_only() {
        let mut r = Renderer::new(caps(false), true);
        let t0 = Instant::now();
        let mut out = vec![];
        out.extend(r.on_event(&AgentEvent::TurnStart, t0));
        out.extend(r.on_event(&AgentEvent::Text("Hel".into()), t0));
        out.extend(r.tick(t0 + Duration::from_secs(5)));
        out.extend(r.on_event(&AgentEvent::Text("lo".into()), t0));
        out.extend(r.on_event(
            &AgentEvent::ToolCall {
                id: "1".into(),
                name: "read".into(),
                arguments: json!({"path": "a.rs"}),
                provider: crate::capability::Provider::Tool,
            },
            t0,
        ));
        out.extend(r.on_event(&AgentEvent::Text("Done.".into()), t0));
        out.extend(r.on_event(&AgentEvent::TurnEnd, t0));
        assert_eq!(texts(&out), vec!["send:Hello", "send:Done."]);
    }

    #[test]
    fn editing_channel_streams_then_formats() {
        let mut r = Renderer::new(caps(true), true);
        let t0 = Instant::now();
        let mut out = vec![];
        out.extend(r.on_event(&AgentEvent::TurnStart, t0));
        out.extend(r.on_event(&AgentEvent::Text("**Hi**".into()), t0));
        out.extend(r.tick(t0));
        out.extend(r.on_event(&AgentEvent::Text(" there".into()), t0));
        out.extend(r.tick(t0 + Duration::from_millis(200))); // throttled
        out.extend(r.tick(t0 + Duration::from_secs(2)));
        out.extend(r.on_event(&AgentEvent::TurnEnd, t0 + Duration::from_secs(3)));
        assert_eq!(
            texts(&out),
            vec![
                "send#1:**Hi** ▍",
                "edit#1:**Hi** there ▍",
                "edit#1:**Hi** there"
            ]
        );
        assert!(matches!(
            out.last(),
            Some(OutItem::Edit {
                format: TextFormat::Markdown,
                ..
            })
        ));
    }

    #[test]
    fn tool_progress_is_one_edited_message() {
        let mut r = Renderer::new(caps(true), true);
        let t0 = Instant::now();
        let call = |n: &str| AgentEvent::ToolCall {
            id: n.into(),
            name: "read".into(),
            arguments: json!({"path": n}),
            provider: crate::capability::Provider::Tool,
        };
        let mut out = vec![];
        out.extend(r.on_event(&AgentEvent::TurnStart, t0));
        out.extend(r.on_event(&call("a.rs"), t0));
        out.extend(r.on_event(&call("b.rs"), t0)); // throttled
        out.extend(r.tick(t0 + Duration::from_secs(2)));
        out.extend(r.on_event(&AgentEvent::TurnEnd, t0 + Duration::from_secs(4)));
        let t = texts(&out);
        assert_eq!(t.len(), 3);
        assert!(t[0].starts_with("send#1:⚙"));
        assert!(t[1].starts_with("edit#1:") && t[1].contains("b.rs"));
        assert!(t[2].ends_with("✓ 本轮调用工具 2 次"));
    }

    #[test]
    fn ask_user_becomes_choices() {
        let mut r = Renderer::new(caps(false), true);
        let out = r.on_event(
            &AgentEvent::ToolCall {
                id: "q".into(),
                name: "ask_user".into(),
                arguments: json!({"question": "Which?", "options": ["A", "B"]}),
                provider: crate::capability::Provider::Human,
            },
            Instant::now(),
        );
        assert_eq!(texts(&out), vec!["ask:Which?:A|B"]);
    }

    #[test]
    fn long_answers_are_chunked() {
        let mut r = Renderer::new(caps(false), true);
        let t0 = Instant::now();
        let mut out = vec![];
        for i in 0..30 {
            out.extend(r.on_event(
                &AgentEvent::Text(format!("paragraph {i} {}\n\n", "x".repeat(40))),
                t0,
            ));
        }
        out.extend(r.on_event(&AgentEvent::TurnEnd, t0));
        assert!(out.len() > 3);
        for t in texts(&out) {
            assert!(t.chars().count() <= 400 + 5, "{t}");
        }
    }
}
