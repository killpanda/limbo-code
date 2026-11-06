//! The chat transcript: entries and how each one renders into lines.
//!
//! Rendering is cached per entry (keyed by width and the expand toggles)
//! so a long session does not re-wrap thousands of lines every frame —
//! only entries that changed (the streaming tail) are re-rendered.

use std::time::Instant;

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;

use super::markdown;
use super::text::{ellipsize, str_width, wrap_line};
use super::theme::{SPINNER, Theme, gradient_spans};
use crate::capability::Provider;
use crate::compaction::is_summary_message;
use crate::models::{Message, Role};
use crate::subagents::describe_call;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Ok,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoticeKind {
    Info,
    Warn,
    Error,
    Compaction,
}

#[derive(Debug, Clone)]
pub enum EntryKind {
    User {
        text: String,
    },
    Assistant {
        text: String,
        streaming: bool,
    },
    Thinking {
        text: String,
        live: bool,
        started: Instant,
        secs: Option<f64>,
    },
    Tool {
        id: String,
        name: String,
        /// Who carried the call out; `None` for a tool no longer registered.
        provider: Option<Provider>,
        args: Value,
        status: ToolStatus,
        output: String,
        progress: Vec<String>,
        started: Instant,
        secs: Option<f64>,
    },
    Notice {
        kind: NoticeKind,
        text: String,
    },
}

pub struct Entry {
    pub kind: EntryKind,
    cache: Option<(usize, bool, bool, Vec<Line<'static>>)>,
}

impl Entry {
    pub fn new(kind: EntryKind) -> Self {
        Self { kind, cache: None }
    }
    pub fn touch(&mut self) {
        self.cache = None;
    }
    fn animated(&self) -> bool {
        matches!(
            self.kind,
            EntryKind::Tool {
                status: ToolStatus::Running,
                ..
            } | EntryKind::Thinking { live: true, .. }
        )
    }

    pub fn lines(
        &mut self,
        theme: &Theme,
        width: usize,
        expand_tools: bool,
        show_thinking: bool,
        frame: usize,
    ) -> &[Line<'static>] {
        let fresh = match &self.cache {
            Some((w, e, t, _)) => {
                *w != width || *e != expand_tools || *t != show_thinking || self.animated()
            }
            None => true,
        };
        if fresh {
            let lines = render_entry(&self.kind, theme, width, expand_tools, show_thinking, frame);
            self.cache = Some((width, expand_tools, show_thinking, lines));
        }
        &self.cache.as_ref().unwrap().3
    }
}

fn tool_icon(name: &str) -> &'static str {
    match name {
        "read" => "◇",
        "write" => "✚",
        "edit" => "✎",
        "bash" => "$",
        "grep" | "find" => "⌕",
        "ls" => "▤",
        "task" => "◈",
        _ => "⚙",
    }
}

fn fmt_secs(s: f64) -> String {
    if s < 10.0 {
        format!("{s:.1}s")
    } else if s < 60.0 {
        format!("{s:.0}s")
    } else {
        format!("{}m{:02}s", s as u64 / 60, s as u64 % 60)
    }
}

fn render_entry(
    kind: &EntryKind,
    t: &Theme,
    width: usize,
    expand: bool,
    show_thinking: bool,
    frame: usize,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    match kind {
        EntryKind::User { text } => {
            let bar = Span::styled("▌ ", Style::default().fg(t.user));
            for (i, l) in text.lines().enumerate() {
                let mut spans = vec![bar.clone()];
                if i == 0 {
                    spans.push(Span::styled(
                        "❯ ",
                        Style::default().fg(t.user).add_modifier(Modifier::BOLD),
                    ));
                } else {
                    spans.push(Span::raw("  "));
                }
                spans.push(Span::styled(
                    l.to_string(),
                    Style::default().fg(t.fg).add_modifier(Modifier::BOLD),
                ));
                out.extend(wrap_line(
                    Line::from(spans),
                    width,
                    &[bar.clone(), Span::raw("  ")],
                ));
            }
            if text.is_empty() {
                out.push(Line::from(vec![
                    bar,
                    Span::styled("❯", Style::default().fg(t.user)),
                ]));
            }
        }
        EntryKind::Assistant { text, streaming } => {
            let body = markdown::render(
                text,
                t,
                width.saturating_sub(2).max(10),
                Style::default().fg(t.fg),
            );
            for (i, l) in body.into_iter().enumerate() {
                let mut spans = if i == 0 {
                    vec![Span::styled(
                        "◆ ",
                        Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
                    )]
                } else {
                    vec![Span::raw("  ")]
                };
                spans.extend(l.spans);
                out.push(Line::from(spans));
            }
            if *streaming {
                let cursor = if (frame / 4).is_multiple_of(2) {
                    "▋"
                } else {
                    " "
                };
                match out.last_mut() {
                    Some(last) => last
                        .spans
                        .push(Span::styled(cursor, Style::default().fg(t.accent))),
                    None => out.push(Line::from(vec![
                        Span::styled("◆ ", Style::default().fg(t.accent)),
                        Span::styled(cursor, Style::default().fg(t.accent)),
                    ])),
                }
            }
        }
        EntryKind::Thinking {
            text,
            live,
            started,
            secs,
        } => {
            let dim = Style::default().fg(t.faint).add_modifier(Modifier::ITALIC);
            let header = if *live {
                let mut spans = vec![Span::styled("✻ ", Style::default().fg(t.accent3))];
                spans.extend(gradient_spans(
                    "Thinking…",
                    &[t.accent3, t.accent, t.accent2],
                    frame as f32 * 0.03,
                    Modifier::ITALIC,
                ));
                spans.push(Span::styled(
                    format!("  {}", fmt_secs(started.elapsed().as_secs_f64())),
                    Style::default().fg(t.faint),
                ));
                Line::from(spans)
            } else {
                Line::from(vec![
                    Span::styled("✻ ", Style::default().fg(t.faint)),
                    Span::styled(
                        format!("thought for {}", fmt_secs(secs.unwrap_or(0.0))),
                        dim,
                    ),
                    Span::styled(
                        if show_thinking {
                            ""
                        } else {
                            "  (ctrl+t to show)"
                        },
                        Style::default().fg(t.faint),
                    ),
                ])
            };
            out.push(header);
            let lines: Vec<&str> = text
                .trim()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .collect();
            let shown: Vec<&str> = if show_thinking {
                lines
            } else if *live {
                lines.iter().rev().take(3).rev().copied().collect()
            } else {
                vec![]
            };
            for l in shown {
                out.extend(wrap_line(
                    Line::from(vec![
                        Span::styled("  ┊ ", Style::default().fg(t.faint)),
                        Span::styled(l.to_string(), dim),
                    ]),
                    width,
                    &[Span::styled("  ┊ ", Style::default().fg(t.faint))],
                ));
            }
        }
        EntryKind::Tool {
            name,
            provider,
            args,
            status,
            output,
            progress,
            started,
            secs,
            ..
        } => {
            let (mark, color) = match status {
                ToolStatus::Running => (SPINNER[frame % SPINNER.len()], t.accent),
                ToolStatus::Ok => ("✓", t.ok),
                ToolStatus::Failed => ("✗", t.err),
            };
            let summary = describe_call(name, args);
            let detail = summary
                .strip_prefix(name.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let elapsed = match (status, secs) {
                (ToolStatus::Running, _) => fmt_secs(started.elapsed().as_secs_f64()),
                (_, Some(s)) => fmt_secs(*s),
                _ => String::new(),
            };
            let right = format!(" {mark} {elapsed}");
            let tag = provider
                .map(|p| format!(" · {}", p.label()))
                .unwrap_or_default();
            let label_w = str_width(name) + str_width(&tag) + 4;
            let avail = width.saturating_sub(label_w + str_width(&right) + 2);
            let detail = ellipsize(&detail, avail);
            let pad = width.saturating_sub(label_w + str_width(&detail) + str_width(&right) + 1);
            out.push(Line::from(vec![
                Span::styled(
                    format!("{} ", tool_icon(name)),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    name.clone(),
                    Style::default().fg(t.accent3).add_modifier(Modifier::BOLD),
                ),
                Span::styled(tag, Style::default().fg(t.faint)),
                Span::raw("  "),
                Span::styled(detail, Style::default().fg(t.dim)),
                Span::styled(" ".repeat(pad), Style::default()),
                Span::styled(right, Style::default().fg(color)),
            ]));
            let gutter = Span::styled("  │ ", Style::default().fg(t.faint));
            let mut body: Vec<Line<'static>> = Vec::new();
            for p in progress
                .iter()
                .rev()
                .take(if expand { 50 } else { 4 })
                .rev()
            {
                body.push(Line::from(vec![
                    gutter.clone(),
                    Span::styled(p.clone(), Style::default().fg(t.dim)),
                ]));
            }
            if *status != ToolStatus::Running && !output.is_empty() {
                let lines: Vec<&str> = output.lines().collect();
                let limit = if expand {
                    400
                } else if *status == ToolStatus::Failed || name == "edit" {
                    12
                } else {
                    4
                };
                // edits start with "Edited x." then a diff: show the diff.
                for l in lines.iter().take(limit) {
                    let style = if name == "edit" || name == "write" {
                        if l.starts_with('+') && !l.starts_with("+++") {
                            Style::default().fg(t.ok)
                        } else if l.starts_with('-') && !l.starts_with("---") {
                            Style::default().fg(t.err)
                        } else if l.starts_with("@@") {
                            Style::default().fg(t.accent3)
                        } else {
                            Style::default().fg(t.dim)
                        }
                    } else if *status == ToolStatus::Failed {
                        Style::default().fg(t.err)
                    } else {
                        Style::default().fg(t.dim)
                    };
                    let l = l.replace('\t', "    ");
                    body.extend(wrap_line(
                        Line::from(vec![gutter.clone(), Span::styled(l, style)]),
                        width,
                        std::slice::from_ref(&gutter),
                    ));
                }
                if lines.len() > limit {
                    body.push(Line::from(vec![
                        Span::styled("  ╰ ", Style::default().fg(t.faint)),
                        Span::styled(
                            format!("… {} more lines (ctrl+o to expand)", lines.len() - limit),
                            Style::default().fg(t.faint).add_modifier(Modifier::ITALIC),
                        ),
                    ]));
                }
            }
            out.extend(body);
        }
        EntryKind::Notice { kind, text } => {
            let (icon, color) = match kind {
                NoticeKind::Info => ("●", t.accent),
                NoticeKind::Warn => ("▲", t.warn),
                NoticeKind::Error => ("✗", t.err),
                NoticeKind::Compaction => ("⊙", t.accent3),
            };
            if *kind == NoticeKind::Compaction {
                let label = format!(" {icon} {text} ");
                let side = width.saturating_sub(str_width(&label)) / 2;
                out.push(Line::from(vec![
                    Span::styled("─".repeat(side), Style::default().fg(t.faint)),
                    Span::styled(label, Style::default().fg(color)),
                    Span::styled("─".repeat(side), Style::default().fg(t.faint)),
                ]));
            } else {
                for (i, l) in text.lines().enumerate() {
                    let head = if i == 0 {
                        format!("{icon} ")
                    } else {
                        "  ".into()
                    };
                    out.extend(wrap_line(
                        Line::from(vec![
                            Span::styled(head, Style::default().fg(color)),
                            Span::styled(l.to_string(), Style::default().fg(color)),
                        ]),
                        width,
                        &[Span::raw("  ")],
                    ));
                }
            }
        }
    }
    out
}

/// Rebuild transcript entries from a loaded history.
/// `provider` names who carries a tool out (from the live registry).
pub fn entries_from_history(
    messages: &[Message],
    provider: impl Fn(&str) -> Option<Provider>,
) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    for m in messages {
        match m.role {
            Role::System => {}
            Role::User if is_summary_message(m) => out.push(Entry::new(EntryKind::Notice {
                kind: NoticeKind::Compaction,
                text: "earlier context compacted".into(),
            })),
            Role::User => out.push(Entry::new(EntryKind::User {
                text: m.text().to_string(),
            })),
            Role::Assistant => {
                if let Some(r) = m.reasoning.as_ref().filter(|r| !r.trim().is_empty()) {
                    out.push(Entry::new(EntryKind::Thinking {
                        text: r.clone(),
                        live: false,
                        started: Instant::now(),
                        secs: None,
                    }));
                }
                if !m.text().trim().is_empty() {
                    out.push(Entry::new(EntryKind::Assistant {
                        text: m.text().to_string(),
                        streaming: false,
                    }));
                }
                for c in m.tool_calls.iter().flatten() {
                    out.push(Entry::new(EntryKind::Tool {
                        id: c.id.clone(),
                        name: c.function.name.clone(),
                        provider: provider(&c.function.name),
                        args: c.function.arguments.clone(),
                        status: ToolStatus::Running,
                        output: String::new(),
                        progress: vec![],
                        started: Instant::now(),
                        secs: None,
                    }));
                }
            }
            Role::Tool => {
                let id = m.tool_call_id.clone().unwrap_or_default();
                if let Some(e) = out
                    .iter_mut()
                    .rev()
                    .find(|e| matches!(&e.kind, EntryKind::Tool { id: i, .. } if *i == id))
                    && let EntryKind::Tool { status, output, .. } = &mut e.kind
                {
                    *output = m.text().to_string();
                    let failed = output.starts_with("Error")
                        || output.starts_with("Command failed")
                        || output.starts_with("[session restored");
                    *status = if failed {
                        ToolStatus::Failed
                    } else {
                        ToolStatus::Ok
                    };
                }
            }
        }
    }
    // Calls whose results never arrived.
    for e in &mut out {
        if let EntryKind::Tool { status, .. } = &mut e.kind
            && *status == ToolStatus::Running
        {
            *status = ToolStatus::Failed;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_card_shows_provider() {
        let mut e = Entry::new(EntryKind::Tool {
            id: "1".into(),
            name: "run_code".into(),
            provider: Some(Provider::Program),
            args: serde_json::json!({"code": "print(1)", "description": "count"}),
            status: ToolStatus::Ok,
            output: String::new(),
            progress: vec![],
            started: Instant::now(),
            secs: Some(1.0),
        });
        let theme = super::super::theme::by_name(None);
        let head: String = e.lines(&theme, 80, false, false, 0)[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(head.contains("run_code · program"), "{head}");
    }
}
