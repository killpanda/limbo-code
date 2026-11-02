//! Collapsed pastes and attachment markers for the prompt box.
//!
//! Big pastes collapse into one-line markers (`[粘贴的文本 #1，共 120 行]`)
//! whose content lives here; at submit time markers expand back. Markers
//! without stored content (hand-typed lookalikes) stay literal, so a paste
//! never vanishes silently. Clipboard images and files become `[图片 #n]` /
//! `[文件 #n: name]` markers whose attachment travels with the message as
//! long as the marker survives editing.

use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;

use crate::attachments::attachment_for;
use crate::models::Attachment;

pub const COLLAPSE_LINES: usize = 10;
pub const COLLAPSE_CHARS: usize = 1000;

fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[粘贴的文本 #(\d+)(?:，共 \d+ 行|，\d+ 字符)\]").unwrap())
}

/// Normalize line endings and strip control chars (keep `\n` and `\t`).
pub fn clean(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .chars()
        .filter(|c| *c == '\n' || *c == '\t' || !c.is_control())
        .collect()
}

#[derive(Default)]
pub struct PasteStore {
    pastes: std::collections::HashMap<usize, String>,
    counter: usize,
    attachments: Vec<(String, Attachment)>,
    attach_counter: usize,
}

impl PasteStore {
    /// An oversized paste's marker (small pastes come back unchanged).
    pub fn collapse(&mut self, text: &str) -> String {
        let lines = text.split('\n').count();
        let chars = text.chars().count();
        if lines <= COLLAPSE_LINES && chars <= COLLAPSE_CHARS {
            return text.to_string();
        }
        self.counter += 1;
        self.pastes.insert(self.counter, text.to_string());
        if lines > COLLAPSE_LINES {
            format!("[粘贴的文本 #{}，共 {lines} 行]", self.counter)
        } else {
            format!("[粘贴的文本 #{}，{chars} 字符]", self.counter)
        }
    }

    /// Expand stored markers; returns the text and ids that were not stored.
    pub fn expand(&self, text: &str) -> (String, Vec<usize>) {
        let mut invalid = vec![];
        let out = marker_re()
            .replace_all(text, |c: &regex::Captures| {
                let id: usize = c[1].parse().unwrap_or(0);
                match self.pastes.get(&id) {
                    Some(t) => t.clone(),
                    None => {
                        invalid.push(id);
                        c[0].to_string()
                    }
                }
            })
            .into_owned();
        (out, invalid)
    }

    /// A stored marker ending exactly at byte `col` of `text`: its span.
    pub fn marker_ending_at(&mut self, text: &str, col: usize) -> Option<(usize, usize)> {
        for m in marker_re().captures_iter(text) {
            let whole = m.get(0).unwrap();
            let id: usize = m[1].parse().unwrap_or(0);
            if whole.end() == col && self.pastes.remove(&id).is_some() {
                return Some((whole.start(), whole.end()));
            }
        }
        None
    }

    pub fn add_image(&mut self, path: &Path, ext: &str) -> String {
        self.attach_counter += 1;
        let n = self.attach_counter;
        let marker = format!("[图片 #{n}]");
        self.attachments.push((
            marker.clone(),
            Attachment {
                kind: "image".into(),
                name: format!("剪贴板图片-{n}.{ext}"),
                path: path.display().to_string(),
                mime: Some(format!("image/{ext}")),
            },
        ));
        marker
    }

    pub fn add_file(&mut self, path: &Path) -> String {
        self.attach_counter += 1;
        let n = self.attach_counter;
        let att = attachment_for(path);
        let marker = if att.kind == "image" {
            format!("[图片 #{n}: {}]", att.name)
        } else {
            format!("[文件 #{n}: {}]", att.name)
        };
        self.attachments.push((marker.clone(), att));
        marker
    }

    /// Attachments whose marker survived editing.
    pub fn live_attachments(&self, text: &str) -> Vec<Attachment> {
        self.attachments
            .iter()
            .filter(|(m, _)| text.contains(m.as_str()))
            .map(|(_, a)| a.clone())
            .collect()
    }

    pub fn clear(&mut self) {
        self.pastes.clear();
        self.attachments.clear();
    }
}

/// Whether `/`-leading text is more likely a file path than a command.
pub fn looks_like_path(text: &str) -> bool {
    let token = text.split_whitespace().next().unwrap_or(text);
    token.get(1..).is_some_and(|r| r.contains('/')) || Path::new(&expand_home(token)).exists()
}

fn expand_home(p: &str) -> String {
    match p.strip_prefix("~/") {
        Some(rest) => std::env::var("HOME").map_or(p.to_string(), |h| format!("{h}/{rest}")),
        None => p.to_string(),
    }
}

/// Existing files named by absolute (or `~`) path tokens become attachments.
pub fn attachments_from_text(text: &str) -> Vec<Attachment> {
    let mut seen = std::collections::HashSet::new();
    text.split_whitespace()
        .filter(|t| t.starts_with('/') || t.starts_with('~'))
        .map(expand_home)
        .filter(|p| Path::new(p).is_absolute() && Path::new(p).is_file())
        .filter(|p| seen.insert(p.clone()))
        .map(|p| attachment_for(Path::new(&p)))
        .collect()
}

/// A paste that is just one or more existing file paths (drag & drop).
pub fn dropped_paths(text: &str) -> Option<Vec<std::path::PathBuf>> {
    let t = text.trim();
    if t.is_empty() || t.lines().count() > 20 {
        return None;
    }
    let mut out = vec![];
    for line in t.lines() {
        let l = line.trim().trim_matches(|c| c == '\'' || c == '"');
        let l = l.strip_prefix("file://").unwrap_or(l).replace("\\ ", " ");
        let p = std::path::PathBuf::from(expand_home(&l));
        if !p.is_absolute() || !p.is_file() {
            return None;
        }
        out.push(p);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_and_expand() {
        let mut s = PasteStore::default();
        assert_eq!(s.collapse("short"), "short");
        let big = (0..12)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let m = s.collapse(&big);
        assert_eq!(m, "[粘贴的文本 #1，共 12 行]");
        let (out, bad) = s.expand(&format!("see {m} and [粘贴的文本 #9，共 3 行]"));
        assert!(out.starts_with("see 0\n1"));
        assert_eq!(bad, vec![9]);
        let text = format!("x{m}");
        assert_eq!(s.marker_ending_at(&text, text.len()), Some((1, text.len())));
        assert!(s.expand(&m).1 == vec![1]);
    }

    #[test]
    fn attachments_and_paths() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("pic.png");
        std::fs::write(&f, "x").unwrap();
        let mut s = PasteStore::default();
        let m = s.add_file(&f);
        assert!(m.starts_with("[图片 #1"));
        assert_eq!(s.live_attachments(&format!("look {m}")).len(), 1);
        assert!(s.live_attachments("deleted").is_empty());
        assert!(looks_like_path("/usr/bin/env"));
        assert!(!looks_like_path("/help"));
        assert_eq!(
            attachments_from_text(&format!("open {}", f.display())).len(),
            1
        );
        assert_eq!(
            dropped_paths(&format!("'{}'", f.display())).unwrap(),
            vec![f]
        );
        assert_eq!(clean("a\r\nb\x07"), "a\nb");
    }
}
