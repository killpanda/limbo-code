//! A small multi-line text editor for the prompt box.
//!
//! The text is a `String`; the cursor is a *char* index (not a byte index —
//! slicing a UTF-8 string in the middle of a character would panic, so we
//! convert with `char_indices` whenever we touch the bytes).

use super::text::char_width;

#[derive(Default)]
pub struct Input {
    pub text: String,
    /// Cursor position in chars.
    pub cursor: usize,
    history: Vec<String>,
    history_pos: Option<usize>,
}

impl Input {
    fn byte_at(&self, char_idx: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_idx)
            .map_or(self.text.len(), |(b, _)| b)
    }
    fn len_chars(&self) -> usize {
        self.text.chars().count()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn insert(&mut self, c: char) {
        let b = self.byte_at(self.cursor);
        self.text.insert(b, c);
        self.cursor += 1;
        self.history_pos = None;
    }

    pub fn insert_str(&mut self, s: &str) {
        let s = s.replace("\r\n", "\n").replace('\r', "\n");
        let b = self.byte_at(self.cursor);
        self.text.insert_str(b, &s);
        self.cursor += s.chars().count();
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let b = self.byte_at(self.cursor - 1);
        self.text.remove(b);
        self.cursor -= 1;
    }

    pub fn delete(&mut self) {
        if self.cursor < self.len_chars() {
            let b = self.byte_at(self.cursor);
            self.text.remove(b);
        }
    }

    /// Delete the word before the cursor (Ctrl+W / Alt+Backspace).
    pub fn delete_word(&mut self) {
        let chars: Vec<char> = self.text.chars().collect();
        let mut i = self.cursor;
        while i > 0 && chars[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !chars[i - 1].is_whitespace() {
            i -= 1;
        }
        let (a, b) = (self.byte_at(i), self.byte_at(self.cursor));
        self.text.replace_range(a..b, "");
        self.cursor = i;
    }

    /// Delete to the start of the line (Ctrl+U).
    pub fn kill_line(&mut self) {
        let start = self.line_start(self.cursor);
        let (a, b) = (self.byte_at(start), self.byte_at(self.cursor));
        self.text.replace_range(a..b, "");
        self.cursor = start;
    }

    fn line_start(&self, pos: usize) -> usize {
        let chars: Vec<char> = self.text.chars().collect();
        let mut i = pos;
        while i > 0 && chars[i - 1] != '\n' {
            i -= 1;
        }
        i
    }
    fn line_end(&self, pos: usize) -> usize {
        let chars: Vec<char> = self.text.chars().collect();
        let mut i = pos;
        while i < chars.len() && chars[i] != '\n' {
            i += 1;
        }
        i
    }

    pub fn left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }
    pub fn right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.len_chars());
    }
    pub fn home(&mut self) {
        self.cursor = self.line_start(self.cursor);
    }
    pub fn end(&mut self) {
        self.cursor = self.line_end(self.cursor);
    }
    pub fn on_first_line(&self) -> bool {
        !self.text.chars().take(self.cursor).any(|c| c == '\n')
    }
    pub fn on_last_line(&self) -> bool {
        !self.text.chars().skip(self.cursor).any(|c| c == '\n')
    }

    pub fn up(&mut self) {
        let start = self.line_start(self.cursor);
        if start == 0 {
            return;
        }
        let col = self.cursor - start;
        let prev_start = self.line_start(start - 1);
        self.cursor = (prev_start + col).min(start - 1);
    }
    pub fn down(&mut self) {
        let end = self.line_end(self.cursor);
        if end >= self.len_chars() {
            return;
        }
        let col = self.cursor - self.line_start(self.cursor);
        let next_end = self.line_end(end + 1);
        self.cursor = (end + 1 + col).min(next_end);
    }

    pub fn take(&mut self) -> String {
        let t = std::mem::take(&mut self.text);
        self.cursor = 0;
        if !t.trim().is_empty() && self.history.last() != Some(&t) {
            self.history.push(t.clone());
        }
        self.history_pos = None;
        t
    }

    pub fn set(&mut self, text: String) {
        self.cursor = text.chars().count();
        self.text = text;
    }

    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let pos = match self.history_pos {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(p) => p - 1,
        };
        self.history_pos = Some(pos);
        self.set(self.history[pos].clone());
    }
    pub fn history_next(&mut self) {
        match self.history_pos {
            Some(p) if p + 1 < self.history.len() => {
                self.history_pos = Some(p + 1);
                self.set(self.history[p + 1].clone());
            }
            Some(_) => {
                self.history_pos = None;
                self.set(String::new());
            }
            None => {}
        }
    }

    /// Visual rows for a box `width` columns wide, plus the cursor (row, col).
    pub fn layout(&self, width: usize) -> (Vec<String>, (usize, usize)) {
        let width = width.max(2);
        let mut rows = vec![String::new()];
        let mut w = 0;
        let mut cur = (0, 0);
        for (i, c) in self.text.chars().enumerate() {
            if i == self.cursor {
                cur = (rows.len() - 1, w);
            }
            if c == '\n' {
                rows.push(String::new());
                w = 0;
                continue;
            }
            let cw = char_width(c);
            if w + cw > width {
                rows.push(String::new());
                w = 0;
            }
            rows.last_mut().unwrap().push(c);
            w += cw;
        }
        if self.cursor >= self.len_chars() {
            if w >= width {
                rows.push(String::new());
                w = 0;
            }
            cur = (rows.len() - 1, w);
        }
        (rows, cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_unicode_safely() {
        let mut i = Input::default();
        for c in "你好".chars() {
            i.insert(c);
        }
        i.left();
        i.insert('!');
        assert_eq!(i.text, "你!好");
        i.backspace();
        i.delete();
        assert_eq!(i.text, "你");
    }

    #[test]
    fn up_down_across_lines() {
        let mut i = Input::default();
        i.insert_str("abc\nde");
        i.up();
        assert_eq!(i.cursor, 2);
        i.down();
        assert_eq!(i.cursor, 6);
    }

    #[test]
    fn layout_cursor() {
        let mut i = Input::default();
        i.insert_str("abcdef");
        let (rows, cur) = i.layout(4);
        assert_eq!(rows, vec!["abcd", "ef"]);
        assert_eq!(cur, (1, 2));
    }
}
