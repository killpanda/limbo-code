//! Line wrapping that understands wide (CJK) characters.
//!
//! We wrap ourselves instead of letting `Paragraph` do it so we know the
//! exact number of rows every chat entry takes — needed for scrolling and
//! for caching rendered entries.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

pub fn char_width(c: char) -> usize {
    if c == '\t' { 4 } else { c.width().unwrap_or(0) }
}

pub fn str_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Wrap one styled line to `width` columns. `indent` is a styled prefix
/// repeated on continuation rows (e.g. the gutter of a block quote).
pub fn wrap_line(
    line: Line<'static>,
    width: usize,
    indent: &[Span<'static>],
) -> Vec<Line<'static>> {
    let width = width.max(4);
    let indent_w: usize = indent.iter().map(|s| str_width(&s.content)).sum();
    // Flatten into (char, style) cells.
    let cells: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|s| {
            let st = line.style.patch(s.style);
            s.content
                .chars()
                .map(move |c| (if c == '\t' { ' ' } else { c }, st))
        })
        .collect();
    if cells.iter().map(|(c, _)| char_width(*c)).sum::<usize>() <= width {
        return vec![line];
    }
    let mut rows: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_w = 0;
    let mut first = true;
    for (c, st) in cells {
        let w = char_width(c);
        let limit = if first {
            width
        } else {
            width.saturating_sub(indent_w).max(1)
        };
        if cur_w + w > limit {
            // Prefer to break after the last space (for Latin text).
            let split = cur
                .iter()
                .rposition(|(ch, _)| *ch == ' ')
                .filter(|&p| p > 0 && cur.len() - p < 24);
            let carry: Vec<(char, Style)> = match split {
                Some(p) if c != ' ' => cur.split_off(p + 1),
                _ => Vec::new(),
            };
            rows.push(std::mem::take(&mut cur));
            first = false;
            cur_w = carry.iter().map(|(ch, _)| char_width(*ch)).sum();
            cur = carry;
            if c == ' ' && cur.is_empty() {
                continue; // don't start a row with the breaking space
            }
        }
        cur.push((c, st));
        cur_w += w;
    }
    if !cur.is_empty() || rows.is_empty() {
        rows.push(cur);
    }
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| {
            let mut spans: Vec<Span<'static>> = if i > 0 { indent.to_vec() } else { Vec::new() };
            spans.extend(merge_cells(row));
            Line::from(spans)
        })
        .collect()
}

/// Re-group consecutive cells of the same style into spans.
fn merge_cells(row: Vec<(char, Style)>) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut style: Option<Style> = None;
    for (c, st) in row {
        if style != Some(st) {
            if let Some(s) = style {
                spans.push(Span::styled(std::mem::take(&mut buf), s));
            }
            style = Some(st);
        }
        buf.push(c);
    }
    if let Some(s) = style {
        spans.push(Span::styled(buf, s));
    }
    spans
}

/// Truncate to `max` columns with an ellipsis.
pub fn ellipsize(s: &str, max: usize) -> String {
    if str_width(s) <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0;
    for c in s.chars() {
        let cw = char_width(c);
        if w + cw + 1 > max {
            break;
        }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn wraps_at_spaces() {
        let out = wrap_line(Line::from("hello brave new world"), 12, &[]);
        assert_eq!(text(&out), vec!["hello brave ", "new world"]);
    }

    #[test]
    fn wraps_cjk_by_width() {
        let out = wrap_line(Line::from("你好世界你好"), 8, &[]);
        assert_eq!(text(&out), vec!["你好世界", "你好"]);
    }

    #[test]
    fn indent_on_continuation() {
        let out = wrap_line(Line::from("aaaa bbbb"), 6, &[Span::raw("> ")]);
        assert_eq!(text(&out), vec!["aaaa ", "> bbbb"]);
    }
}
