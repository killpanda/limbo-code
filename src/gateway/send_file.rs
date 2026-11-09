//! `send_file`: deliver a local file to the person on the chat this session
//! is attached to. Registered only in gateway sessions.
//!
//! Effect `mutate` (it reaches outside the machine, and plan mode blocks
//! it), recovery `inspect` (a crash must not send the file twice), provider
//! `channel`.

use std::path::Path;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::render::OutItem;
use crate::capability::Provider;
use crate::channels::OutboundMessage;
use crate::effect::{Effect, Recovery};
use crate::models::ToolResult;
use crate::tools::{Tool, ToolContext, arg_str, resolve_path};

pub struct SendFileTool {
    pub out: mpsc::UnboundedSender<OutItem>,
    /// Display name of the chat ("Telegram" / "微信").
    pub channel: &'static str,
    pub max_bytes: u64,
}

#[async_trait]
impl Tool for SendFileTool {
    fn name(&self) -> &'static str {
        "send_file"
    }
    fn description(&self) -> String {
        format!(
            "Send a local file (report, image, log, patch) to the user in this {} chat. \
             Use it when the user asks for a file, or when a result is too long to read as a message.",
            self.channel
        )
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to send (relative to the workdir, or absolute)."},
                "caption": {"type": "string", "description": "Optional one-line caption."},
            },
            "required": ["path"],
        })
    }
    fn effect(&self) -> Effect {
        Effect::Mutate
    }
    fn recovery(&self) -> Recovery {
        Recovery::Inspect
    }
    fn provider(&self) -> Provider {
        Provider::Channel
    }
    fn program_callable(&self) -> bool {
        false
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let Some(raw) = arg_str(&args, "path").filter(|p| !p.trim().is_empty()) else {
            return ToolResult::err("path must be a non-empty string");
        };
        let path = resolve_path(&ctx.workdir, raw);
        check(&path, self.max_bytes)
            .map(|size| {
                let caption = arg_str(&args, "caption").map(String::from);
                let _ = self.out.send(OutItem::Send {
                    msg: OutboundMessage::File {
                        path: path.clone(),
                        caption,
                    },
                    slot: None,
                });
                ToolResult::ok(format!(
                    "Queued {} ({size} bytes) for delivery to the user's {} chat.",
                    path.display(),
                    self.channel
                ))
            })
            .unwrap_or_else(ToolResult::err)
    }
}

fn check(path: &Path, max: u64) -> Result<u64, String> {
    let meta =
        std::fs::metadata(path).map_err(|e| format!("cannot send {}: {e}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a file", path.display()));
    }
    if meta.len() > max {
        return Err(format!(
            "{} is {} bytes; the chat accepts at most {max}",
            path.display(),
            meta.len()
        ));
    }
    Ok(meta.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queues_existing_files_only() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("r.txt"), "report").unwrap();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let tool = SendFileTool {
            out: tx,
            channel: "Telegram",
            max_bytes: 1024,
        };
        let ctx = ToolContext::simple(dir.path());
        let r = tool
            .run(json!({"path": "r.txt", "caption": "日报"}), &ctx)
            .await;
        assert!(r.success, "{:?}", r.error);
        assert!(matches!(
            rx.try_recv(),
            Ok(OutItem::Send { msg: OutboundMessage::File { caption: Some(c), .. }, .. }) if c == "日报"
        ));
        let r = tool.run(json!({"path": "missing.txt"}), &ctx).await;
        assert!(!r.success);
        assert!(rx.try_recv().is_err());
    }
}
