//! Markdown → styled terminal lines (pulldown-cmark events → ratatui spans).

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::text::{str_width, wrap_line};
use super::theme::Theme;

struct Renderer<'t> {
    theme: &'t Theme,
    width: usize,
    out: Vec<Line<'static>>,
    cur: Vec<Span<'static>>,
    styles: Vec<Style>,
    /// Stack of list counters (`None` = bullet list).
    lists: Vec<Option<u64>>,
    quote_depth: usize,
    in_code: Option<String>,
    code_buf: String,
    pending_item: Option<String>,
    link_url: Option<String>,
    table_row: Vec<String>,
    table_cell: String,
    in_table: bool,
}

impl<'t> Renderer<'t> {
    fn style(&self) -> Style {
        self.styles.last().copied().unwrap_or_default()
    }

    fn prefix(&self) -> Vec<Span<'static>> {
        let mut p = Vec::new();
        for _ in 0..self.quote_depth {
            p.push(Span::styled("│ ", Style::default().fg(self.theme.accent3)));
        }
        if !self.lists.is_empty() {
            p.push(Span::raw("  ".repeat(self.lists.len())));
        }
        p
    }

    fn flush(&mut self) {
        if self.cur.is_empty() && self.pending_item.is_none() {
            return;
        }
        let mut spans = Vec::new();
        for _ in 0..self.quote_depth {
            spans.push(Span::styled("│ ", Style::default().fg(self.theme.accent3)));
        }
        let mut indent = self.prefix();
        if let Some(marker) = self.pending_item.take() {
            let depth = self.lists.len().saturating_sub(1);
            spans.push(Span::raw("  ".repeat(depth)));
            let w = str_width(&marker);
            spans.push(Span::styled(marker, Style::default().fg(self.theme.accent)));
            indent.push(Span::raw(" ".repeat(w)));
        } else if !self.lists.is_empty() {
            spans.push(Span::raw("  ".repeat(self.lists.len())));
        }
        spans.append(&mut self.cur);
        // `prefix()` already includes the quote gutter; drop duplicates.
        let indent: Vec<Span<'static>> = indent;
        self.out
            .extend(wrap_line(Line::from(spans), self.width, &indent));
    }

    fn blank(&mut self) {
        if self.out.last().is_some_and(|l| !l.spans.is_empty()) {
            self.out.push(Line::default());
        }
    }

    fn push_text(&mut self, text: &str) {
        let st = self.style();
        if let Some(url) = &self.link_url
            && text == url.as_str()
        {
            self.cur.push(Span::styled(text.to_string(), st));
            return;
        }
        let mut first = true;
        for part in text.split('\n') {
            if !first {
                self.flush();
            }
            first = false;
            if !part.is_empty() {
                self.cur.push(Span::styled(part.to_string(), st));
            }
        }
    }

    fn code_block(&mut self, lang: &str, code: &str) {
        let t = self.theme;
        let bg = t.panel;
        let label = if lang.is_empty() {
            "code".to_string()
        } else {
            lang.to_string()
        };
        let inner = self.width.saturating_sub(2).max(10);
        self.out.push(Line::from(vec![
            Span::styled("╭─ ", Style::default().fg(t.faint)),
            Span::styled(
                label.clone(),
                Style::default().fg(t.accent3).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    " {}",
                    "─".repeat(inner.saturating_sub(str_width(&label) + 3))
                ),
                Style::default().fg(t.faint),
            ),
        ]));
        for l in code.trim_end_matches('\n').split('\n') {
            let line = Line::from(vec![
                Span::styled("│ ", Style::default().fg(t.faint)),
                Span::styled(l.replace('\t', "    "), Style::default().fg(t.code).bg(bg)),
            ]);
            self.out.extend(wrap_line(
                line,
                self.width,
                &[Span::styled("│ ", Style::default().fg(t.faint))],
            ));
        }
        self.out.push(Line::from(Span::styled(
            format!("╰{}", "─".repeat(inner)),
            Style::default().fg(t.faint),
        )));
    }

    fn event(&mut self, ev: Event) {
        let t = self.theme;
        match ev {
            Event::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    self.flush();
                    self.blank();
                    let (marker, color) = match level {
                        HeadingLevel::H1 => ("▍", t.accent2),
                        HeadingLevel::H2 => ("▍", t.accent),
                        _ => ("▸ ", t.accent3),
                    };
                    self.cur
                        .push(Span::styled(marker.to_string(), Style::default().fg(color)));
                    self.styles
                        .push(Style::default().fg(color).add_modifier(Modifier::BOLD));
                }
                Tag::Paragraph => {}
                Tag::BlockQuote(_) => {
                    self.flush();
                    self.quote_depth += 1;
                    self.styles
                        .push(self.style().fg(t.dim).add_modifier(Modifier::ITALIC));
                }
                Tag::CodeBlock(kind) => {
                    self.flush();
                    self.in_code = Some(match kind {
                        CodeBlockKind::Fenced(l) => l.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    });
                    self.code_buf.clear();
                }
                Tag::List(start) => {
                    self.flush();
                    self.lists.push(start);
                }
                Tag::Item => {
                    self.flush();
                    let marker = match self.lists.last_mut() {
                        Some(Some(n)) => {
                            let m = format!("{n}. ");
                            *n += 1;
                            m
                        }
                        _ => {
                            ["• ", "◦ ", "▪ "][(self.lists.len().saturating_sub(1)) % 3].to_string()
                        }
                    };
                    self.pending_item = Some(marker);
                }
                Tag::Emphasis => self
                    .styles
                    .push(self.style().add_modifier(Modifier::ITALIC)),
                Tag::Strong => self
                    .styles
                    .push(self.style().add_modifier(Modifier::BOLD).fg(t.fg)),
                Tag::Strikethrough => self
                    .styles
                    .push(self.style().add_modifier(Modifier::CROSSED_OUT)),
                Tag::Link { dest_url, .. } => {
                    self.link_url = Some(dest_url.to_string());
                    self.styles
                        .push(self.style().fg(t.accent).add_modifier(Modifier::UNDERLINED));
                }
                Tag::Table(_) => {
                    self.flush();
                    self.in_table = true;
                }
                Tag::TableHead | Tag::TableRow => self.table_row.clear(),
                Tag::TableCell => self.table_cell.clear(),
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Heading(_) => {
                    self.styles.pop();
                    self.flush();
                }
                TagEnd::Paragraph => {
                    self.flush();
                    if self.lists.is_empty() {
                        self.blank();
                    }
                }
                TagEnd::BlockQuote(_) => {
                    self.flush();
                    self.styles.pop();
                    self.quote_depth = self.quote_depth.saturating_sub(1);
                    self.blank();
                }
                TagEnd::CodeBlock => {
                    let lang = self.in_code.take().unwrap_or_default();
                    let code = std::mem::take(&mut self.code_buf);
                    self.code_block(&lang, &code);
                    self.blank();
                }
                TagEnd::List(_) => {
                    self.flush();
                    self.lists.pop();
                    if self.lists.is_empty() {
                        self.blank();
                    }
                }
                TagEnd::Item => self.flush(),
                TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                    self.styles.pop();
                }
                TagEnd::Link => {
                    self.styles.pop();
                    self.link_url = None;
                }
                TagEnd::TableCell => {
                    let c = std::mem::take(&mut self.table_cell);
                    self.table_row.push(c.trim().to_string());
                }
                TagEnd::TableHead | TagEnd::TableRow => {
                    let is_head = matches!(tag, TagEnd::TableHead);
                    let row = std::mem::take(&mut self.table_row);
                    let mut spans = vec![Span::styled("│ ", Style::default().fg(t.faint))];
                    for (i, cell) in row.iter().enumerate() {
                        if i > 0 {
                            spans.push(Span::styled(" │ ", Style::default().fg(t.faint)));
                        }
                        let st = if is_head {
                            Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
                        } else {
                            Style::default().fg(t.fg)
                        };
                        spans.push(Span::styled(cell.clone(), st));
                    }
                    self.out
                        .extend(wrap_line(Line::from(spans), self.width, &[Span::raw("  ")]));
                    if is_head {
                        self.out.push(Line::from(Span::styled(
                            "├".to_string() + &"─".repeat(self.width.saturating_sub(1).min(60)),
                            Style::default().fg(t.faint),
                        )));
                    }
                }
                TagEnd::Table => {
                    self.in_table = false;
                    self.blank();
                }
                _ => {}
            },
            Event::Text(text) => {
                if self.in_code.is_some() {
                    self.code_buf.push_str(&text);
                } else if self.in_table {
                    self.table_cell.push_str(&text);
                } else {
                    self.push_text(&text);
                }
            }
            Event::Code(code) => {
                if self.in_table {
                    self.table_cell.push_str(&code);
                } else {
                    self.cur.push(Span::styled(
                        format!(" {code} "),
                        Style::default().fg(t.code).bg(t.panel_hi),
                    ));
                }
            }
            Event::SoftBreak => self.cur.push(Span::raw(" ")),
            Event::HardBreak => self.flush(),
            Event::Rule => {
                self.flush();
                self.out.push(Line::from(Span::styled(
                    "─".repeat(self.width.min(80)),
                    Style::default().fg(t.faint),
                )));
            }
            Event::TaskListMarker(done) => {
                self.pending_item = Some(if done { "☑ ".into() } else { "☐ ".into() });
            }
            Event::Html(h) | Event::InlineHtml(h) => self.push_text(&h),
            _ => {}
        }
    }
}

/// Render markdown into wrapped lines `width` columns wide.
pub fn render(md: &str, theme: &Theme, width: usize, base: Style) -> Vec<Line<'static>> {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_TASKLISTS);
    let mut r = Renderer {
        theme,
        width,
        out: Vec::new(),
        cur: Vec::new(),
        styles: vec![base],
        lists: Vec::new(),
        quote_depth: 0,
        in_code: None,
        code_buf: String::new(),
        pending_item: None,
        link_url: None,
        table_row: Vec::new(),
        table_cell: String::new(),
        in_table: false,
    };
    for ev in Parser::new_ext(md, opts) {
        r.event(ev);
    }
    // An unterminated code fence while streaming: show what we have.
    if let Some(lang) = r.in_code.take() {
        let code = std::mem::take(&mut r.code_buf);
        r.code_block(&lang, &code);
    }
    r.flush();
    while r.out.last().is_some_and(|l| l.spans.is_empty()) {
        r.out.pop();
    }
    r.out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::NEON;

    fn plain(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn lists_and_code() {
        let out = plain(&render(
            "# Title\n\n- a\n- b\n\n```rs\nfn x() {}\n```",
            &NEON,
            40,
            Style::default(),
        ));
        assert!(out[0].contains("Title"));
        assert!(out.iter().any(|l| l == "• a"));
        assert!(out.iter().any(|l| l.contains("rs")));
        assert!(out.iter().any(|l| l.contains("fn x() {}")));
    }

    #[test]
    fn numbered_list() {
        let out = plain(&render("1. one\n2. two", &NEON, 40, Style::default()));
        assert_eq!(out, vec!["1. one", "2. two"]);
    }
}
