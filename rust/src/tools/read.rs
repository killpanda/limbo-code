//! `read`: text files (with offset/limit) and images (sent as attachments).

use std::path::Path;

use async_trait::async_trait;
use serde_json::{Value, json};

use super::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head};
use super::{Tool, ToolContext, arg_i64, arg_str, resolve_path};
use crate::models::{Attachment, ToolResult};

const MAX_FILE_SIZE: u64 = 50 * 1024 * 1024;
const MAX_IMAGE_SIZE: u64 = 20 * 1024 * 1024;

pub struct ReadTool;

/// Sniff magic bytes for jpg/png/gif/webp/bmp.
pub fn detect_image_mime(path: &Path) -> Option<&'static str> {
    use std::io::Read;
    let mut head = [0u8; 16];
    let n = std::fs::File::open(path).ok()?.read(&mut head).ok()?;
    let h = &head[..n];
    if h.starts_with(&[0xff, 0xd8, 0xff]) {
        return (h.get(3) != Some(&0xf7)).then_some("image/jpeg");
    }
    if h.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if h.starts_with(b"GIF") {
        return Some("image/gif");
    }
    if h.len() >= 12 && &h[..4] == b"RIFF" && &h[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if h.starts_with(b"BM") {
        return Some("image/bmp");
    }
    None
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }
    fn description(&self) -> String {
        format!(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are \
             sent as attachments. For text files, output is truncated to the first {DEFAULT_MAX_LINES} lines or \
             {}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, \
             continue with offset until complete.",
            DEFAULT_MAX_BYTES / 1024
        )
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "offset": {"type": "integer", "description": "Line number to start from (1-indexed, optional)"},
                "limit": {"type": "integer", "description": "Maximum lines to read (optional)"},
            },
            "required": ["path"],
        })
    }
    fn replay_safe(&self) -> bool {
        true
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let raw = arg_str(&args, "path").unwrap_or("").to_string();
        let target = resolve_path(&ctx.workdir, &raw);
        let offset = arg_i64(&args, "offset");
        let limit = arg_i64(&args, "limit");
        // File I/O is blocking: run it on tokio's blocking thread pool so the
        // async runtime (and the UI) stay responsive.
        tokio::task::spawn_blocking(move || read_file(&raw, &target, offset, limit))
            .await
            .unwrap_or_else(|e| ToolResult::err(format!("read panicked: {e}")))
    }
}

fn read_file(raw: &str, target: &Path, offset: Option<i64>, limit: Option<i64>) -> ToolResult {
    if !target.exists() {
        return ToolResult::err(format!("File not found: {raw}"));
    }
    if !target.is_file() {
        return ToolResult::err(format!("Not a file: {raw}"));
    }
    let size = match std::fs::metadata(target) {
        Ok(m) => m.len(),
        Err(e) => return ToolResult::err(format!("Could not read file: {e}")),
    };
    if let Some(mime) = detect_image_mime(target) {
        if size > MAX_IMAGE_SIZE {
            return ToolResult::err(format!(
                "Image too large ({size} bytes). Maximum size is {MAX_IMAGE_SIZE} bytes."
            ));
        }
        return ToolResult {
            success: true,
            output: Some(format!("Read image file [{mime}]")),
            attachments: Some(vec![Attachment {
                kind: "image".into(),
                name: target
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                path: target.display().to_string(),
                mime: Some(mime.into()),
            }]),
            error: None,
        };
    }
    if size > MAX_FILE_SIZE {
        return ToolResult::err(format!(
            "File too large ({size} bytes). Maximum size is {MAX_FILE_SIZE} bytes."
        ));
    }
    if offset.is_some_and(|o| o < 1) {
        return ToolResult::err("offset must be a positive integer.");
    }
    if limit.is_some_and(|l| l < 1) {
        return ToolResult::err("limit must be a positive integer.");
    }
    let bytes = match std::fs::read(target) {
        Ok(b) => b,
        Err(e) => return ToolResult::err(format!("Could not read file: {e}")),
    };
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let total = lines.len();
    let start = offset.map_or(0, |o| (o - 1) as usize).min(total);
    let end = limit.map_or(total, |l| (start + l as usize).min(total));
    let selected: String = lines[start..end].concat();
    let t = truncate_head(&selected, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    let mut out = t.content;
    if t.truncated {
        let first = offset.unwrap_or(1);
        let last = first + t.output_lines as i64 - 1;
        out.push_str(&format!(
            "\n[Output truncated: showing lines {first}-{last} of {total}. Use offset/limit to read more.]"
        ));
    }
    ToolResult::ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(dir: &Path) -> ToolContext {
        ToolContext::simple(dir.to_path_buf())
    }

    #[tokio::test]
    async fn reads_with_offset_and_limit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "1\n2\n3\n4\n").unwrap();
        let r = ReadTool
            .run(
                json!({"path": "a.txt", "offset": 2, "limit": 2}),
                &ctx(dir.path()),
            )
            .await;
        assert_eq!(r.output.unwrap(), "2\n3\n");
    }

    #[tokio::test]
    async fn missing_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let r = ReadTool
            .run(json!({"path": "nope"}), &ctx(dir.path()))
            .await;
        assert!(!r.success);
    }

    #[tokio::test]
    async fn images_become_attachments() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.bin"), b"\x89PNG\r\n\x1a\n0000").unwrap();
        let r = ReadTool
            .run(json!({"path": "x.bin"}), &ctx(dir.path()))
            .await;
        assert_eq!(r.attachments.unwrap()[0].mime.as_deref(), Some("image/png"));
    }
}
