//! How user attachments reach the model.
//!
//! Images become multimodal blocks only for vision models; everything else
//! degrades to text: small text files inline, the rest as path references
//! the model can open with `read`. Nothing is silently dropped.

use std::path::Path;

use crate::models::Attachment;

/// Text files at or below this size are inlined into the user message.
pub const ATTACHMENT_INLINE_MAX_BYTES: u64 = 50_000;

pub fn read_inline_text(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() > ATTACHMENT_INLINE_MAX_BYTES {
        return None;
    }
    String::from_utf8(std::fs::read(path).ok()?).ok()
}

/// `(content, images)`: the typed text plus notes for non-image attachments.
pub fn build_user_content(input: &str, attachments: &[Attachment], vision: bool) -> (String, Vec<Attachment>) {
    let mut images = Vec::new();
    let mut notes = Vec::new();
    for a in attachments {
        let path = Path::new(&a.path);
        if a.kind == "image" {
            if vision && path.exists() {
                images.push(a.clone());
            } else if vision {
                notes.push(format!("[图片 {} 文件已不存在：{}]", a.name, path.display()));
            } else {
                notes.push(format!(
                    "[图片 {} 已保存到 {}；当前模型不支持图像输入，无法直接查看]",
                    a.name,
                    path.display()
                ));
            }
            continue;
        }
        match read_inline_text(path) {
            Some(text) => notes.push(format!("文件 {} 的内容：\n```\n{text}\n```", a.name)),
            None => notes.push(format!("[文件 {} 位于 {}，可用 read 工具查看]", a.name, path.display())),
        }
    }
    if notes.is_empty() {
        return (input.to_string(), images);
    }
    (format!("{input}\n\n{}", notes.join("\n")), images)
}

/// Guess an attachment from a path: images by extension, files otherwise.
pub fn attachment_for(path: &Path) -> Attachment {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let mime = match ext.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    };
    Attachment {
        kind: if mime.is_some() { "image" } else { "file" }.into(),
        name,
        path: path.display().to_string(),
        mime: mime.map(String::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degrades_without_vision() {
        let dir = tempfile::tempdir().unwrap();
        let txt = dir.path().join("a.txt");
        std::fs::write(&txt, "hello").unwrap();
        let img = dir.path().join("p.png");
        std::fs::write(&img, [0u8; 4]).unwrap();
        let (c, imgs) = build_user_content("hi", &[attachment_for(&txt), attachment_for(&img)], false);
        assert!(imgs.is_empty());
        assert!(c.contains("文件 a.txt 的内容"));
        assert!(c.contains("不支持图像输入"));
        let (_, imgs) = build_user_content("hi", &[attachment_for(&img)], true);
        assert_eq!(imgs.len(), 1);
    }
}
