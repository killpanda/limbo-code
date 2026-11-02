//! Tools the model can call, and the registry that dispatches them.
//!
//! Like the Python version there are no safety fences: absolute paths are
//! honored, and `bash` runs whatever it is given. The only gate is plan
//! mode, which hides and blocks every tool that can change things.
//!
//! 🦀 Rust notes
//! - [`Tool`] is an async trait. `#[async_trait]` rewrites `async fn` in a
//!   trait into a method returning a boxed future, which lets us store
//!   different tools side by side as `Arc<dyn Tool>`.
//! - `Arc` = atomically reference-counted pointer: many owners, freed when
//!   the last one goes away. Tools are shared between concurrent calls.

pub mod bash;
pub mod edit;
pub mod find;
pub mod grep;
pub mod ls;
pub mod read;
pub mod truncate;
pub mod write;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::models::ToolResult;

/// Everything a tool call may need besides its arguments.
#[derive(Clone)]
pub struct ToolContext {
    pub workdir: PathBuf,
    /// Cancelled when the user presses Esc; long-running tools (bash,
    /// subagents) watch it and stop early.
    pub cancel: CancellationToken,
    /// The id of the call being run (subagents use it for their session file).
    pub call_id: String,
    /// The calling agent's session file (subagents live next to it).
    pub session_path: Option<PathBuf>,
    /// Where live progress goes (subagent activity shown in the tool card).
    pub events: Option<crate::agent::EventTx>,
    /// Plan mode: subagents are forced read-only too.
    pub plan_mode: bool,
}

impl ToolContext {
    /// A bare context (tests, one-off calls).
    #[cfg(test)]
    pub fn simple(workdir: impl Into<PathBuf>) -> Self {
        Self {
            workdir: workdir.into(),
            cancel: CancellationToken::new(),
            call_id: "call".into(),
            session_path: None,
            events: None,
            plan_mode: false,
        }
    }
    pub fn progress(&self, line: impl Into<String>) {
        if let Some(tx) = &self.events {
            let _ = tx.send(crate::agent::AgentEvent::ToolProgress {
                id: self.call_id.clone(),
                line: line.into(),
            });
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> String;
    /// JSON Schema of the arguments.
    fn parameters(&self) -> Value;
    /// Read-only tools are re-run after a crash; everything else is not.
    fn replay_safe(&self) -> bool {
        false
    }
    /// Whether the tool can change files or state (blocked in plan mode).
    fn mutates(&self) -> bool {
        !self.replay_safe()
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult;
}

/// The tools available in a session.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
    pub plan_mode: bool,
}

pub const PLAN_MODE_INSTRUCTION: &str = "Plan mode is on: you are read-only. Explore with read/grep/find/ls; \
edit, write and bash are blocked. Finish by presenting a concrete, step-by-step plan. The user turns plan \
mode off (/plan off) before anything is changed.";

impl ToolRegistry {
    /// The seven core tools (bash optional).
    pub fn core(bash_enabled: bool) -> Self {
        let mut tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(read::ReadTool),
            Arc::new(write::WriteTool),
            Arc::new(edit::EditTool),
        ];
        if bash_enabled {
            tools.push(Arc::new(bash::BashTool));
        }
        tools.push(Arc::new(grep::GrepTool));
        tools.push(Arc::new(find::FindTool));
        tools.push(Arc::new(ls::LsTool));
        Self {
            tools,
            plan_mode: false,
        }
    }

    /// Only the read-only tools (used by read-only subagents).
    pub fn read_only() -> Self {
        let mut r = Self::core(false);
        r.tools.retain(|t| t.replay_safe());
        r
    }

    pub fn add(&mut self, tool: Arc<dyn Tool>) {
        self.tools.retain(|t| t.name() != tool.name());
        self.tools.push(tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    pub fn names(&self) -> Vec<&'static str> {
        self.visible().map(|t| t.name()).collect()
    }

    fn visible(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools
            .iter()
            .filter(move |t| !self.plan_mode || !t.mutates())
    }

    /// OpenAI-style function definitions for the request.
    pub fn definitions(&self) -> Vec<Value> {
        self.visible()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {"name": t.name(), "description": t.description(), "parameters": t.parameters()},
                })
            })
            .collect()
    }

    /// Run one call. Unknown tools and plan-mode blocks become error results
    /// the model can read and react to.
    pub async fn execute(&self, name: &str, args: Value, ctx: &ToolContext) -> ToolResult {
        let Some(tool) = self.get(name) else {
            return ToolResult::err(format!("Unknown tool: {name}"));
        };
        if self.plan_mode && tool.mutates() {
            return ToolResult::err(format!(
                "Plan mode is on (read-only): {name} is blocked. Keep exploring with read-only tools, then \
                 present the plan; the user leaves plan mode with /plan off before anything is changed."
            ));
        }
        if let Some(err) = args.get("parse_error").and_then(Value::as_str) {
            return ToolResult::err(format!(
                "Tool arguments were not valid JSON ({err}). Re-issue the call with valid JSON arguments."
            ));
        }
        tool.run(args, ctx).await
    }
}

// -- shared helpers ----------------------------------------------------------

/// Resolve a path argument: `~` expands, relative paths are under the workdir.
pub fn resolve_path(workdir: &Path, raw: &str) -> PathBuf {
    let expanded = if raw == "~" {
        dirs::home_dir().unwrap_or_default()
    } else if let Some(rest) = raw.strip_prefix("~/") {
        dirs::home_dir().unwrap_or_default().join(rest)
    } else {
        PathBuf::from(raw)
    };
    let joined = if expanded.is_absolute() {
        expanded
    } else {
        workdir.join(expanded)
    };
    normalize(&joined)
}

/// Lexically clean `.` and `..` (symlinks are left alone, like the Python
/// version's non-strict resolve for paths that may not exist yet).
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// A display path: relative to the workdir when inside it.
pub fn display_path(workdir: &Path, p: &Path) -> String {
    p.strip_prefix(workdir)
        .map(|r| r.display().to_string())
        .unwrap_or_else(|_| p.display().to_string())
}

pub fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

/// Integers sometimes arrive as strings or floats — accept all three.
pub fn arg_i64(args: &Value, key: &str) -> Option<i64> {
    match args.get(key)? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn arg_bool(args: &Value, key: &str) -> bool {
    match args.get(key) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => matches!(s.as_str(), "true" | "1" | "yes"),
        _ => false,
    }
}

/// Per-file locks so parallel write/edit calls to the same file never
/// interleave (different files still run in parallel).
pub fn mutation_lock(path: &Path) -> Arc<tokio::sync::Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>> = OnceLock::new();
    let key = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let mut map = LOCKS.get_or_init(Default::default).lock().unwrap();
    map.entry(key).or_default().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative_and_dotdot() {
        let wd = Path::new("/work/proj");
        assert_eq!(
            resolve_path(wd, "src/../a.rs"),
            PathBuf::from("/work/proj/a.rs")
        );
        assert_eq!(resolve_path(wd, "/etc/hosts"), PathBuf::from("/etc/hosts"));
    }

    #[tokio::test]
    async fn plan_mode_blocks_mutations() {
        let mut reg = ToolRegistry::core(true);
        reg.plan_mode = true;
        assert!(!reg.names().contains(&"write"));
        let ctx = ToolContext::simple("/tmp");
        let r = reg
            .execute("write", json!({"path": "x", "content": ""}), &ctx)
            .await;
        assert!(!r.success);
        assert!(r.error.unwrap().contains("Plan mode"));
    }
}
