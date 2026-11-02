//! Output truncation: keep the head (file reads) or the tail (command output)
//! within a line *and* byte budget, never splitting a line or a UTF-8 char.

pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
pub const MAX_OUTPUT_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Truncation {
    pub content: String,
    pub truncated: bool,
    pub total_lines: usize,
    pub output_lines: usize,
    /// 1-indexed first kept line (>1 only for tail truncation).
    pub start_line: usize,
}

fn split_lines(s: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = s.split('\n').collect();
    if s.ends_with('\n') {
        lines.pop();
    }
    lines
}

/// Largest prefix of `s` that fits in `max` bytes on a char boundary.
pub fn head_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn tail_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut start = s.len() - max;
    while !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

pub fn truncate_head(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = split_lines(content);
    let total = lines.len();
    if total <= max_lines && content.len() <= max_bytes {
        return Truncation {
            content: content.into(),
            truncated: false,
            total_lines: total,
            output_lines: total,
            start_line: 1,
        };
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0;
    for line in &lines {
        if kept.len() >= max_lines {
            break;
        }
        let cost = line.len() + usize::from(!kept.is_empty());
        if bytes + cost > max_bytes {
            if kept.is_empty() {
                kept.push(head_bytes(line, max_bytes));
            }
            break;
        }
        kept.push(line);
        bytes += cost;
    }
    Truncation {
        output_lines: kept.len(),
        content: kept.join("\n"),
        truncated: true,
        total_lines: total,
        start_line: 1,
    }
}

pub fn truncate_tail(content: &str, max_lines: usize, max_bytes: usize) -> Truncation {
    let lines = split_lines(content);
    let total = lines.len();
    if total <= max_lines && content.len() <= max_bytes {
        return Truncation {
            content: content.into(),
            truncated: false,
            total_lines: total,
            output_lines: total,
            start_line: 1,
        };
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut bytes = 0;
    for line in lines.iter().rev() {
        if kept.len() >= max_lines {
            break;
        }
        let cost = line.len() + usize::from(!kept.is_empty());
        if bytes + cost > max_bytes {
            if kept.is_empty() {
                kept.push(tail_bytes(line, max_bytes));
            }
            break;
        }
        kept.push(line);
        bytes += cost;
    }
    kept.reverse();
    let n = kept.len();
    Truncation {
        content: kept.join("\n"),
        truncated: true,
        total_lines: total,
        output_lines: n,
        start_line: total - n + 1,
    }
}

/// Cap `s` at `limit` bytes and append a notice.
pub fn truncate_output(s: String, limit: usize) -> String {
    if s.len() <= limit {
        return s;
    }
    format!("{}\n[Output truncated.]", head_bytes(&s, limit))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_by_lines() {
        let t = truncate_head("a\nb\nc\n", 2, 100);
        assert!(t.truncated);
        assert_eq!(t.content, "a\nb");
        assert_eq!(t.total_lines, 3);
    }

    #[test]
    fn tail_by_bytes_keeps_whole_lines() {
        let t = truncate_tail("aaaa\nbbbb\ncccc", 100, 9);
        assert_eq!(t.content, "bbbb\ncccc");
        assert_eq!(t.start_line, 2);
    }

    #[test]
    fn utf8_safe() {
        let t = truncate_head("你好世界", 10, 4);
        assert_eq!(t.content, "你");
    }
}
