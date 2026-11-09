//! Model markdown → what a chat platform renders: Telegram's HTML subset or
//! plain text (WeChat), plus splitting long answers into message-sized
//! chunks without cutting a code block in half.
//!
//! 🦀 Rust notes
//! - `pulldown_cmark::Parser` is an iterator of `Event`s (Start(tag),
//!   Text, End(tag), …); the converters fold it into a `String`.
//! - `match` on `(&Event, mode)` pairs keeps both dialects in one walker.

use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};

use super::MarkdownDialect;

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Render `md` in `dialect`.
pub fn render(md: &str, dialect: MarkdownDialect) -> String {
    let html = dialect == MarkdownDialect::TelegramHtml;
    let mut out = String::new();
    // Stack of list counters (None = bullet list).
    let mut lists: Vec<Option<u64>> = Vec::new();
    let mut link_url: Vec<String> = Vec::new();
    // The open code block has a `<code class=..>` inside its `<pre>`.
    let mut pre_code = false;
    let opts = Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TABLES;
    for ev in Parser::new_ext(md, opts) {
        match ev {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {}
                Tag::Heading { .. } => {
                    if html {
                        out.push_str("<b>");
                    }
                }
                Tag::BlockQuote(_) => {
                    if html {
                        out.push_str("<blockquote>");
                    } else {
                        out.push_str("> ");
                    }
                }
                Tag::CodeBlock(kind) => {
                    if html {
                        match kind {
                            CodeBlockKind::Fenced(lang) if !lang.is_empty() => {
                                pre_code = true;
                                out.push_str(&format!(
                                    "<pre><code class=\"language-{}\">",
                                    escape_html(&lang)
                                ))
                            }
                            _ => {
                                pre_code = false;
                                out.push_str("<pre>")
                            }
                        }
                    }
                }
                Tag::List(start) => lists.push(start),
                Tag::Item => {
                    let depth = lists.len().saturating_sub(1);
                    out.push_str(&"  ".repeat(depth));
                    match lists.last_mut() {
                        Some(Some(n)) => {
                            out.push_str(&format!("{n}. "));
                            *n += 1;
                        }
                        _ => out.push_str("• "),
                    }
                }
                Tag::Emphasis => {
                    if html {
                        out.push_str("<i>");
                    }
                }
                Tag::Strong => {
                    if html {
                        out.push_str("<b>");
                    }
                }
                Tag::Strikethrough => {
                    if html {
                        out.push_str("<s>");
                    }
                }
                Tag::Link { dest_url, .. } => {
                    if html {
                        out.push_str(&format!("<a href=\"{}\">", escape_html(&dest_url)));
                    }
                    link_url.push(dest_url.to_string());
                }
                Tag::Image { dest_url, .. } => link_url.push(dest_url.to_string()),
                Tag::TableCell => out.push_str("| "),
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph => out.push_str("\n\n"),
                TagEnd::Heading(_) => {
                    if html {
                        out.push_str("</b>");
                    }
                    out.push_str("\n\n");
                }
                TagEnd::BlockQuote(_) => {
                    trim_end_newlines(&mut out);
                    if html {
                        out.push_str("</blockquote>");
                    }
                    out.push_str("\n\n");
                }
                TagEnd::CodeBlock => {
                    if out.ends_with('\n') {
                        out.pop();
                    }
                    if html {
                        out.push_str(if pre_code { "</code></pre>" } else { "</pre>" });
                    }
                    out.push_str("\n\n");
                }
                TagEnd::List(_) => {
                    lists.pop();
                    if lists.is_empty() {
                        trim_end_newlines(&mut out);
                        out.push_str("\n\n");
                    }
                }
                TagEnd::Item => {
                    trim_end_newlines(&mut out);
                    out.push('\n');
                }
                TagEnd::Emphasis => {
                    if html {
                        out.push_str("</i>");
                    }
                }
                TagEnd::Strong => {
                    if html {
                        out.push_str("</b>");
                    }
                }
                TagEnd::Strikethrough => {
                    if html {
                        out.push_str("</s>");
                    }
                }
                TagEnd::Link => {
                    let url = link_url.pop().unwrap_or_default();
                    if html {
                        out.push_str("</a>");
                    } else if !url.is_empty() && !out.ends_with(&url) {
                        out.push_str(&format!(" ({url})"));
                    }
                }
                TagEnd::Image => {
                    let url = link_url.pop().unwrap_or_default();
                    out.push_str(&format!(" ({url})"));
                }
                TagEnd::TableCell => out.push(' '),
                TagEnd::TableRow | TagEnd::TableHead => out.push_str("|\n"),
                TagEnd::Table => out.push('\n'),
                _ => {}
            },
            Event::Text(t) => {
                if html {
                    out.push_str(&escape_html(&t));
                } else {
                    out.push_str(&t);
                }
            }
            Event::Code(t) => {
                if html {
                    out.push_str(&format!("<code>{}</code>", escape_html(&t)));
                } else {
                    out.push_str(&t);
                }
            }
            Event::SoftBreak | Event::HardBreak => out.push('\n'),
            Event::Rule => out.push_str("──────\n\n"),
            Event::Html(t) | Event::InlineHtml(t) => {
                if html {
                    out.push_str(&escape_html(&t));
                } else {
                    out.push_str(&t);
                }
            }
            Event::TaskListMarker(done) => out.push_str(if done { "☑ " } else { "☐ " }),
            _ => {}
        }
    }
    out.trim_end().to_string()
}

fn trim_end_newlines(s: &mut String) {
    while s.ends_with('\n') {
        s.pop();
    }
}

/// Split markdown source into chunks of at most `max` characters, at
/// paragraph boundaries first, then lines; a code block that has to be cut
/// is closed and reopened so each chunk renders on its own.
pub fn split(md: &str, max: usize) -> Vec<String> {
    let max = max.max(64);
    // Blocks: runs of lines separated by blank lines, fences kept whole.
    let mut blocks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_fence = false;
    for line in md.lines() {
        let fence = line.trim_start().starts_with("```");
        if fence {
            in_fence = !in_fence;
        }
        if !in_fence && !fence && line.trim().is_empty() {
            if !cur.is_empty() {
                blocks.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
    }
    if !cur.is_empty() {
        blocks.push(cur);
    }
    let mut pieces: Vec<String> = Vec::new();
    for b in blocks {
        if b.chars().count() <= max {
            pieces.push(b);
        } else {
            pieces.extend(split_block(&b, max));
        }
    }
    // Pack pieces into chunks.
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        let extra = if cur.is_empty() { 0 } else { 2 };
        if !cur.is_empty() && cur.chars().count() + extra + p.chars().count() > max {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(&p);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// One oversized block: by lines; inside a fence, every piece gets its own
/// opening and closing fence line.
fn split_block(block: &str, max: usize) -> Vec<String> {
    let first = block.lines().next().unwrap_or("");
    let fenced = first.trim_start().starts_with("```");
    let open = if fenced {
        first.to_string()
    } else {
        String::new()
    };
    let close = if fenced { "```" } else { "" };
    let overhead = if fenced {
        open.chars().count() + close.len() + 2
    } else {
        0
    };
    let budget = max.saturating_sub(overhead).max(16);
    let body_lines: Vec<&str> = if fenced {
        let mut v: Vec<&str> = block.lines().skip(1).collect();
        if v.last().is_some_and(|l| l.trim_start().starts_with("```")) {
            v.pop();
        }
        v
    } else {
        block.lines().collect()
    };
    let mut out = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.is_empty() {
            return;
        }
        if fenced {
            out.push(format!("{open}\n{cur}\n{close}"));
        } else {
            out.push(cur.clone());
        }
        cur.clear();
    };
    for line in body_lines {
        // A single line longer than the budget is cut by characters.
        let mut rest: Vec<char> = line.chars().collect();
        loop {
            let take = rest.len().min(budget);
            let piece: String = rest.drain(..take).collect();
            let extra = if cur.is_empty() { 0 } else { 1 };
            if !cur.is_empty() && cur.chars().count() + extra + piece.chars().count() > budget {
                flush(&mut cur, &mut out);
            }
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(&piece);
            if rest.is_empty() {
                break;
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telegram_html_subset() {
        let html = render(
            "# Title\n\nSome **bold** and `a<b>` and [link](https://x.y).\n\n- one\n- two\n\n```rust\nlet a = 1 < 2;\n```",
            MarkdownDialect::TelegramHtml,
        );
        assert!(html.starts_with("<b>Title</b>"));
        assert!(html.contains("<b>bold</b>"));
        assert!(html.contains("<code>a&lt;b&gt;</code>"));
        assert!(html.contains("<a href=\"https://x.y\">link</a>"));
        assert!(html.contains("• one\n• two"));
        assert!(html.contains("<pre><code class=\"language-rust\">let a = 1 &lt; 2;</code></pre>"));
    }

    #[test]
    fn plain_text_drops_markup() {
        let t = render(
            "## Steps\n\n1. **Read** it\n2. Edit [docs](https://d.e)\n\n```\ncode\n```",
            MarkdownDialect::Plain,
        );
        assert_eq!(t, "Steps\n\n1. Read it\n2. Edit docs (https://d.e)\n\ncode");
    }

    #[test]
    fn split_keeps_paragraphs_and_fences() {
        let md = format!(
            "{}\n\n{}\n\n```\n{}\n```",
            "a".repeat(50),
            "b".repeat(50),
            (0..40)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let chunks = split(&md, 120);
        assert!(chunks.len() >= 3);
        for c in &chunks {
            assert!(c.chars().count() <= 120, "{c}");
            assert_eq!(c.matches("```").count() % 2, 0, "unbalanced fence: {c}");
        }
        assert!(chunks[0].starts_with("aaaa"));
    }

    #[test]
    fn short_text_is_one_chunk() {
        assert_eq!(split("hello\n\nworld", 100), vec!["hello\n\nworld"]);
    }
}
