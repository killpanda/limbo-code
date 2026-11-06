//! Mouse text selection. Mouse capture (needed for wheel scrolling) stops the
//! terminal from selecting text itself, so we do it: a left-button drag marks
//! a span of screen cells inside the region where it started (the chat or the
//! prompt box), the span is drawn reversed, and on release its text is read
//! back from the rendered buffer and copied to the clipboard.

use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect};
use ratatui::style::Modifier;
use unicode_width::UnicodeWidthStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    /// The region the drag started in; both ends are clamped to it.
    pub region: Rect,
    pub anchor: Position,
    pub head: Position,
}

impl Selection {
    pub fn new(region: Rect, at: Position) -> Self {
        let p = clamp(region, at);
        Self {
            region,
            anchor: p,
            head: p,
        }
    }

    pub fn extend(&mut self, to: Position) {
        self.head = clamp(self.region, to);
    }

    pub fn is_empty(&self) -> bool {
        self.anchor == self.head
    }

    /// Start and end in reading order (row first, then column).
    fn ordered(&self) -> (Position, Position) {
        let (a, b) = (self.anchor, self.head);
        if (a.y, a.x) <= (b.y, b.x) {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Selected column range (inclusive) of each selected row.
    fn rows(&self) -> impl Iterator<Item = (u16, u16, u16)> + '_ {
        let (s, e) = self.ordered();
        let r = self.region;
        let last = r.right().saturating_sub(1);
        (s.y..=e.y).map(move |y| {
            let x0 = if y == s.y { s.x } else { r.x };
            let x1 = if y == e.y { e.x } else { last };
            (y, x0, x1)
        })
    }
}

fn clamp(r: Rect, p: Position) -> Position {
    Position {
        x: p.x.clamp(r.x, r.right().saturating_sub(1).max(r.x)),
        y: p.y.clamp(r.y, r.bottom().saturating_sub(1).max(r.y)),
    }
}

/// Draw the selection reversed.
pub fn highlight(buf: &mut Buffer, sel: &Selection) {
    if sel.is_empty() {
        return;
    }
    let area = buf.area;
    for (y, x0, x1) in sel.rows() {
        for x in x0..=x1 {
            if area.contains(Position { x, y }) {
                buf[(x, y)].modifier.toggle(Modifier::REVERSED);
            }
        }
    }
}

/// The selected text, one line per row, trailing blanks trimmed.
pub fn extract(buf: &Buffer, sel: &Selection) -> String {
    let area = buf.area;
    let mut lines: Vec<String> = Vec::new();
    for (y, x0, x1) in sel.rows() {
        if y < area.top() || y >= area.bottom() {
            continue;
        }
        let mut line = String::new();
        // Walk from the row start so a wide character's hidden second cell
        // (reset to a blank by ratatui) is skipped, not copied as a space.
        let mut x = sel.region.x.max(area.left());
        let end = x1.min(area.right().saturating_sub(1));
        while x <= end {
            let sym = buf[(x, y)].symbol();
            let w = sym.width().max(1) as u16;
            // Take a cell that starts in the range, or a wide one covering
            // its first column.
            if x >= x0 || x + w > x0 {
                line.push_str(sym);
            }
            x += w;
        }
        lines.push(line.trim_end().to_string());
    }
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Style;

    fn buf(rows: &[&str]) -> Buffer {
        let mut b = Buffer::empty(Rect::new(0, 0, 10, rows.len() as u16));
        for (y, r) in rows.iter().enumerate() {
            b.set_string(0, y as u16, r, Style::default());
        }
        b
    }

    #[test]
    fn single_row() {
        let b = buf(&["hello world"]);
        let mut s = Selection::new(b.area, Position::new(0, 0));
        s.extend(Position::new(4, 0));
        assert_eq!(extract(&b, &s), "hello");
    }

    #[test]
    fn backwards_multi_row() {
        let b = buf(&["abc", "defgh", "ijk"]);
        let mut s = Selection::new(b.area, Position::new(1, 2));
        s.extend(Position::new(1, 0));
        assert_eq!(extract(&b, &s), "bc\ndefgh\nij");
    }

    #[test]
    fn wide_chars_not_padded() {
        let b = buf(&["你好世界"]);
        let mut s = Selection::new(b.area, Position::new(0, 0));
        s.extend(Position::new(9, 0));
        assert_eq!(extract(&b, &s), "你好世界");
        // Starting on the hidden half still takes the whole character.
        let mut s = Selection::new(b.area, Position::new(3, 0));
        s.extend(Position::new(5, 0));
        assert_eq!(extract(&b, &s), "好世");
    }

    #[test]
    fn clamped_to_region() {
        let b = buf(&["|abcdef|", "|ghijkl|"]);
        let region = Rect::new(1, 0, 6, 2);
        let mut s = Selection::new(region, Position::new(0, 0));
        s.extend(Position::new(9, 1));
        assert_eq!(extract(&b, &s), "abcdef\nghijkl");
    }

    #[test]
    fn highlight_toggles_reversed() {
        let mut b = buf(&["abc"]);
        let mut s = Selection::new(b.area, Position::new(0, 0));
        s.extend(Position::new(1, 0));
        highlight(&mut b, &s);
        assert!(b[(0, 0)].modifier.contains(Modifier::REVERSED));
        assert!(b[(1, 0)].modifier.contains(Modifier::REVERSED));
        assert!(!b[(2, 0)].modifier.contains(Modifier::REVERSED));
    }
}
