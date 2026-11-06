//! Tools the model can call, and the registry that dispatches them.
//!
//! Like the Python version there are no safety fences: absolute paths are
//! honored, and `bash` runs whatever it is given. The only gates are
//! admission policies (`admission.rs`): plan mode, which hides and blocks
//! every tool that can change things, and subagent profile restrictions.
//!
//! 🦀 Rust notes
//! - [`Tool`] is an async trait. `#[async_trait]` rewrites `async fn` in a
//!   trait into a method returning a boxed future, which lets us store
//!   different tools side by side as `Arc<dyn Tool>`.
//! - `Arc` = atomically reference-counted pointer: many owners, freed when
//!   the last one goes away. Tools are shared between concurrent calls.

pub mod ask_user;
pub mod bash;
pub mod edit;
pub mod find;
pub mod grep;
pub mod ls;
pub mod memory;
pub mod read;
pub mod run_code;
pub mod session_search;
pub mod skill_manage;
pub mod truncate;
pub mod web_fetch;
pub mod web_search;
pub mod write;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::admission::{self, ActionRequest, PlanPolicy, Policy, Verdict};
use crate::effect::{Effect, Recovery};
use crate::models::ToolResult;
use crate::posture::Posture;

/// Everything a tool call may need besides its arguments.
#[derive(Clone)]
pub struct ToolContext {
    pub workdir: PathBuf,
    /// Cancelled when the user presses Esc; long-running tools (bash,
    /// subagents) watch it and stop early.
    pub cancel: CancellationToken,
    /// The id of the call being run (subagents use it for their session file).
    pub call_id: String,
    /// Where live progress goes (subagent activity shown in the tool card).
    pub events: Option<crate::agent::EventTx>,
    /// The registry running this call (`run_code` dispatches through it).
    pub registry: Option<ToolRegistry>,
}

impl ToolContext {
    /// A bare context (tests, one-off calls).
    #[cfg(test)]
    pub fn simple(workdir: impl Into<PathBuf>) -> Self {
        Self {
            workdir: workdir.into(),
            cancel: CancellationToken::new(),
            call_id: "call".into(),
            events: None,
            registry: None,
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
    /// What running the tool does (see [`crate::effect`]). Unknown tools
    /// are treated as arbitrary execution.
    fn effect(&self) -> Effect {
        Effect::Exec
    }
    /// What resume does with a call a crash interrupted.
    fn recovery(&self) -> Recovery {
        self.effect().default_recovery()
    }
    /// Re-run after a crash (derived from [`Tool::recovery`]).
    fn replay_safe(&self) -> bool {
        self.recovery() == Recovery::Replay
    }
    /// Whether the tool can change files or state (blocked in plan mode;
    /// derived from [`Tool::effect`]).
    fn mutates(&self) -> bool {
        self.effect().mutates()
    }
    /// Whether a `run_code` program may call it. Tools that drive live
    /// session state (the task board) stay native tools beside `run_code`.
    fn program_callable(&self) -> bool {
        true
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult;
}

/// Tools only daily mode registers.
pub const DAILY_TOOLS: [&str; 5] = [
    "web_fetch",
    "memory",
    "session_search",
    "ask_user",
    "skill_manage",
];

/// The tools available in a session.
#[derive(Clone)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
    /// Policy, capability set and presentation (`posture.rs`). Change the
    /// capability set through [`ToolRegistry::set_daily`], which also swaps
    /// the daily tools.
    pub posture: Posture,
    /// Admission policies in front of every call (subagent profiles:
    /// read-only, no-edit, allow-list, path claims); plan mode adds
    /// [`PlanPolicy`] ahead of them.
    pub policies: Vec<Arc<dyn Policy>>,
    /// Set when the `memory` tool ran (daily memory nudge bookkeeping).
    pub memory_written: Arc<AtomicBool>,
    /// Where `session_search` looks.
    pub session_dir: PathBuf,
}

pub const PLAN_MODE_INSTRUCTION: &str = "Plan mode is on: you are read-only. Explore with read/grep/find/ls; \
edit, write and bash are blocked. Finish by presenting a concrete, step-by-step plan. The user turns plan \
mode off (/plan off) before anything is changed.";

impl ToolRegistry {
    /// The seven core tools (bash optional) plus `run_code`.
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
        tools.push(Arc::new(run_code::RunCodeTool));
        Self {
            tools,
            posture: Posture::default(),
            policies: Vec::new(),
            memory_written: Arc::new(AtomicBool::new(false)),
            session_dir: crate::config::limbo_home().join("sessions"),
        }
    }

    /// The full catalog for a config: core tools plus `web_search` when the
    /// search provider has a key.
    pub fn for_config(config: &crate::config::Config) -> Self {
        let mut r = Self::core(config.tools.bash_enabled);
        if let Some(provider) = web_search::build_provider(config) {
            r.add(Arc::new(web_search::WebSearchTool::new(
                provider,
                config.web_search.count,
            )));
        }
        r
    }

    pub fn plan_mode(&self) -> bool {
        self.posture.plan()
    }
    pub fn daily_mode(&self) -> bool {
        self.posture.daily()
    }
    /// Code Mode: the model sees only `run_code` (plus non-programmable tools).
    pub fn code_mode(&self) -> bool {
        self.posture.code()
    }

    /// Switch to `posture`, swapping the daily tools when the capability
    /// set changes.
    pub fn set_posture(&mut self, posture: Posture, config: &crate::config::Config) {
        if posture.daily() != self.daily_mode() {
            self.set_daily(posture.daily(), config);
        }
        self.posture = posture;
    }

    /// Add or remove the daily tools.
    pub fn set_daily(&mut self, enabled: bool, config: &crate::config::Config) {
        self.posture = self.posture.with_daily(enabled);
        self.tools.retain(|t| !DAILY_TOOLS.contains(&t.name()));
        if !enabled {
            return;
        }
        let d = &config.daily;
        self.tools.push(Arc::new(web_fetch::WebFetchTool {
            max_chars: d.fetch_max_chars,
            timeout: d.fetch_timeout,
        }));
        if d.memory {
            self.tools.push(Arc::new(memory::MemoryTool {
                store: crate::memory::MemoryStore::new(
                    crate::memory::memory_dir(),
                    d.user_chars,
                    d.notes_chars,
                ),
            }));
        }
        self.tools.push(Arc::new(session_search::SessionSearchTool {
            dir: self.session_dir.clone(),
        }));
        self.tools.push(Arc::new(ask_user::AskUserTool));
        self.tools.push(Arc::new(skill_manage::SkillManageTool));
    }

    pub fn add(&mut self, tool: Arc<dyn Tool>) {
        self.tools.retain(|t| t.name() != tool.name());
        self.tools.push(tool);
    }

    pub fn remove(&mut self, name: &str) {
        if name != "run_code" {
            self.tools.retain(|t| t.name() != name);
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name).cloned()
    }

    /// Model-facing tool names.
    #[cfg(test)]
    pub fn names(&self) -> Vec<&'static str> {
        self.model_facing().map(|t| t.name()).collect()
    }

    /// Every registered tool name (programmable ones included).
    pub fn all_names(&self) -> Vec<&'static str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// The active policies, in order: plan mode first, then the registry's own.
    fn active_policies(&self) -> impl Iterator<Item = &dyn Policy> {
        let plan: Option<&dyn Policy> = self.plan_mode().then_some(&PlanPolicy);
        plan.into_iter()
            .chain(self.policies.iter().map(|p| p.as_ref()))
    }

    fn allowed(&self, t: &Arc<dyn Tool>) -> bool {
        self.active_policies().all(|p| p.visible(t.as_ref()))
    }

    fn model_facing(&self) -> impl Iterator<Item = &Arc<dyn Tool>> {
        self.tools.iter().filter(move |t| {
            let shown = if self.code_mode() {
                t.name() == "run_code" || !t.program_callable()
            } else {
                t.name() != "run_code"
            };
            shown && self.allowed(t)
        })
    }

    fn definition(t: &Arc<dyn Tool>) -> Value {
        json!({
            "type": "function",
            "function": {"name": t.name(), "description": t.description(), "parameters": t.parameters()},
        })
    }

    /// OpenAI-style function definitions for the request.
    pub fn definitions(&self) -> Vec<Value> {
        self.model_facing().map(Self::definition).collect()
    }

    /// Everything a `run_code` program may call (the SDK section).
    pub fn program_definitions(&self) -> Vec<Value> {
        self.tools
            .iter()
            .filter(|t| t.name() != "run_code" && t.program_callable() && self.allowed(t))
            .map(Self::definition)
            .collect()
    }

    /// Run one call: resolve → admit → run. Unknown tools and denials
    /// become error results the model can read and react to.
    pub async fn execute(&self, name: &str, args: Value, ctx: &ToolContext) -> ToolResult {
        let Some(tool) = self.get(name) else {
            return ToolResult::err(format!("Unknown tool: {name}"));
        };
        let req = ActionRequest {
            tool: tool.as_ref(),
            args: &args,
        };
        if let Verdict::Deny(reason) = admission::admit(self.active_policies(), &req) {
            return ToolResult::err(reason);
        }
        if let Some(err) = args.get("parse_error").and_then(Value::as_str) {
            return ToolResult::err(format!(
                "Tool arguments were not valid JSON ({err}). Re-issue the call with valid JSON arguments."
            ));
        }
        if name == "memory" {
            self.memory_written.store(true, Ordering::SeqCst);
        }
        let mut ctx = ctx.clone();
        if ctx.registry.is_none() {
            ctx.registry = Some(self.clone());
        }
        tool.run(args, &ctx).await
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

    /// The effect table: replay and plan-mode gating follow from it.
    #[test]
    fn core_tool_effects() {
        let reg = ToolRegistry::core(true);
        let row = |n: &str| {
            let t = reg.get(n).unwrap();
            (t.effect(), t.replay_safe(), t.mutates())
        };
        for n in ["read", "grep", "find", "ls"] {
            assert_eq!(row(n), (Effect::Read, true, false), "{n}");
        }
        assert_eq!(row("write"), (Effect::Write, false, true));
        assert_eq!(row("edit"), (Effect::Mutate, false, true));
        assert_eq!(row("bash"), (Effect::Exec, false, true));
        assert_eq!(row("run_code"), (Effect::Session, false, false));
    }

    #[tokio::test]
    async fn plan_mode_blocks_mutations() {
        let mut reg = ToolRegistry::core(true);
        reg.posture = reg.posture.with_plan(true);
        assert!(!reg.names().contains(&"write"));
        let ctx = ToolContext::simple("/tmp");
        let r = reg
            .execute("write", json!({"path": "x", "content": ""}), &ctx)
            .await;
        assert!(!r.success);
        assert!(r.error.unwrap().contains("Plan mode"));
    }
}
