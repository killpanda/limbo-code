//! `bash`: run a shell command in the workdir.
//!
//! The command runs in its own process group so a timeout or an Esc
//! interrupt kills the whole tree, not just the shell. Output is collected
//! while it streams; once it outgrows the inline budget everything is also
//! spilled to a temp file whose path is shown to the model.
//!
//! 🦀 `tokio::select!` waits on several futures at once and runs the branch
//! of whichever finishes first — here: the process exiting, the timeout,
//! or the cancel token firing.

use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};

use super::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail};
use super::{Tool, ToolContext, arg_str};
use crate::models::ToolResult;

pub struct BashTool;

pub const INTERRUPTED_MSG: &str = "已被用户打断（ESC）";

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }
    fn effect(&self) -> crate::effect::Effect {
        crate::effect::Effect::Exec
    }
    fn description(&self) -> String {
        format!(
            "Execute a bash command in the current working directory. Returns stdout and stderr. Output is \
             truncated to the last {DEFAULT_MAX_LINES} lines or {}KB (whichever is hit first). If truncated, the \
             full output is saved to a temp file and its path is shown. Note: bash is not sandboxed — it can \
             access files outside the working directory. For multi-line scripts, write the script to a file with \
             the write tool and execute that file (e.g. python script.py) instead of using heredocs.",
            DEFAULT_MAX_BYTES / 1024
        )
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
            },
            "required": ["command"],
        })
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let command = arg_str(&args, "command").unwrap_or("").to_string();
        if command.is_empty() {
            return ToolResult::err("No command provided.");
        }
        let timeout = match args.get("timeout") {
            None | Some(Value::Null) => None,
            Some(v) => match v
                .as_f64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
            {
                Some(t) if t.is_finite() && t > 0.0 => Some(Duration::from_secs_f64(t)),
                Some(_) => {
                    return ToolResult::err(
                        "Invalid timeout: must be a positive number of seconds",
                    );
                }
                None => return ToolResult::err(format!("Invalid timeout: {v}")),
            },
        };
        run_command(&command, ctx, timeout).await
    }
}

async fn read_all(mut r: impl AsyncRead + Unpin) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        match r.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    buf
}

fn kill_tree(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        // SAFETY: killpg only sends a signal; a stale pid fails harmlessly.
        unsafe {
            libc::killpg(pid as i32, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    let _ = pid;
}

fn spill(output: &str) -> Option<PathBuf> {
    let name = format!("limbo-output-{}.log", crate::models::new_id());
    let path = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&path).ok()?;
    f.write_all(output.as_bytes()).ok()?;
    Some(path)
}

pub async fn run_command(
    command: &str,
    ctx: &ToolContext,
    timeout: Option<Duration>,
) -> ToolResult {
    let shell = if std::path::Path::new("/bin/bash").exists() {
        "/bin/bash"
    } else {
        "bash"
    };
    let mut cmd = tokio::process::Command::new(shell);
    cmd.arg("-c")
        .arg(command)
        .current_dir(&ctx.workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    cmd.process_group(0);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return ToolResult::err(format!("Execution failed: {e}")),
    };
    let pid = child.id();
    let out_task = tokio::spawn(read_all(child.stdout.take().unwrap()));
    let err_task = tokio::spawn(read_all(child.stderr.take().unwrap()));

    enum End {
        Exited(Option<i32>),
        TimedOut,
        Interrupted,
    }
    let sleeper = async {
        match timeout {
            Some(t) => tokio::time::sleep(t).await,
            None => std::future::pending::<()>().await,
        }
    };
    let end = tokio::select! {
        status = child.wait() => End::Exited(status.ok().and_then(|s| s.code())),
        _ = sleeper => End::TimedOut,
        _ = ctx.cancel.cancelled() => End::Interrupted,
    };
    if !matches!(end, End::Exited(_)) {
        kill_tree(pid);
        let _ = tokio::time::timeout(Duration::from_secs(10), child.wait()).await;
    }
    // A grandchild that escaped the group could keep the pipes open forever:
    // bound the wait for the readers.
    let grab = |t: tokio::task::JoinHandle<Vec<u8>>| async move {
        tokio::time::timeout(Duration::from_secs(30), t)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_default()
    };
    let stdout = String::from_utf8_lossy(&grab(out_task).await).replace("\r\n", "\n");
    let stderr = String::from_utf8_lossy(&grab(err_task).await).replace("\r\n", "\n");

    let mut output = stdout;
    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("[stderr]\n");
        output.push_str(&stderr);
    }
    let t = truncate_tail(&output, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES);
    if t.truncated {
        let mut hint = format!(
            "[Showing lines {}-{} of {}",
            t.start_line, t.total_lines, t.total_lines
        );
        match spill(&output) {
            Some(p) => hint.push_str(&format!(". Full output: {}]", p.display())),
            None => hint.push_str("; full output could not be saved to a temp file]"),
        }
        output = format!("{}\n\n{hint}", t.content);
    }
    let prefix = |msg: &str, out: &str| {
        if out.is_empty() {
            msg.to_string()
        } else {
            format!("{msg}\n{out}")
        }
    };
    match end {
        End::Interrupted => ToolResult {
            success: false,
            output: Some(prefix(INTERRUPTED_MSG, &output)),
            error: Some(INTERRUPTED_MSG.into()),
            attachments: None,
            end_turn: false,
        },
        End::TimedOut => {
            let msg = format!(
                "Command timed out after {}s and was killed.",
                timeout.unwrap().as_secs_f64()
            );
            ToolResult {
                success: false,
                output: Some(prefix(&msg, &output)),
                error: Some(msg),
                attachments: None,
                end_turn: false,
            }
        }
        End::Exited(code) if code != Some(0) => {
            let msg = format!(
                "Command failed with exit code {}.",
                code.map_or("?".into(), |c| c.to_string())
            );
            ToolResult {
                success: false,
                output: Some(prefix(&msg, &output)),
                error: Some(msg),
                attachments: None,
                end_turn: false,
            }
        }
        End::Exited(_) => ToolResult::ok(output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolContext {
        ToolContext::simple(std::env::temp_dir())
    }

    #[tokio::test]
    async fn captures_stdout_stderr_and_exit_code() {
        let r = BashTool
            .run(json!({"command": "echo hi; echo oops >&2; exit 3"}), &ctx())
            .await;
        assert!(!r.success);
        let out = r.output.unwrap();
        assert!(out.starts_with("Command failed with exit code 3."));
        assert!(out.contains("hi\n"));
        assert!(out.contains("[stderr]\noops"));
    }

    #[tokio::test]
    async fn timeout_kills() {
        let r = BashTool
            .run(json!({"command": "sleep 5", "timeout": 0.2}), &ctx())
            .await;
        assert!(r.error.unwrap().contains("timed out"));
    }

    #[tokio::test]
    async fn cancel_interrupts() {
        let c = ctx();
        let token = c.cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            token.cancel();
        });
        let r = BashTool.run(json!({"command": "sleep 5"}), &c).await;
        assert_eq!(r.error.as_deref(), Some(INTERRUPTED_MSG));
    }
}
