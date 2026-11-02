//! Tool hooks a subagent's registry gets from its profile.
//!
//! - [`ReadOnlyHook`] — `tools: read_only`: state-changing tools are
//!   blocked, the subagent's own task tools stay callable.
//! - [`NoEditHook`] — `tools: no_edit` (verifier): no `edit`/`write`.
//! - [`ClaimHook`] — writers on a shared workdir: `edit`/`write` refuse a
//!   path another live task claimed in `scope_paths`.
//! - [`AllowListHook`] — cache-parity fork children see the main tool
//!   list but may only call their profile's tools.

use std::path::PathBuf;
use std::sync::Weak;

use serde_json::Value;

use super::supervisor::Supervisor;
use super::tools::SUBAGENT_TOOL_NAMES;
use crate::tools::{Tool, ToolHook};

pub struct ReadOnlyHook;

impl ToolHook for ReadOnlyHook {
    fn before_tool(&self, tool: &dyn Tool, _args: &Value) -> Result<(), String> {
        if SUBAGENT_TOOL_NAMES.contains(&tool.name()) || !tool.mutates() {
            return Ok(());
        }
        Err(format!(
            "This subagent is read-only: {} is blocked. Report what should change in your finish() summary instead.",
            tool.name()
        ))
    }
}

pub struct NoEditHook;

impl ToolHook for NoEditHook {
    fn before_tool(&self, tool: &dyn Tool, _args: &Value) -> Result<(), String> {
        if matches!(tool.name(), "edit" | "write" | "run_code") {
            return Err(format!(
                "This subagent does not edit files: {} is blocked. Report the fix in your finish() summary instead.",
                tool.name()
            ));
        }
        Ok(())
    }
}

pub struct AllowListHook {
    pub allowed: Vec<String>,
}

impl ToolHook for AllowListHook {
    fn before_tool(&self, tool: &dyn Tool, _args: &Value) -> Result<(), String> {
        if self.allowed.is_empty() || self.allowed.iter().any(|a| a == tool.name()) {
            Ok(())
        } else {
            Err(format!(
                "{} is not available to this subagent; it may only call: {}.",
                tool.name(),
                self.allowed.join(", ")
            ))
        }
    }
}

/// `fnmatch`-style match, where `dir/**` also covers `dir` and below.
pub fn matches(rel: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|p| {
        let p = p.trim().trim_start_matches("./");
        if p.is_empty() {
            return false;
        }
        if glob_match(p, rel) {
            return true;
        }
        let base = p.trim_end_matches("/**").trim_end_matches("/*");
        base != p && (rel == base || rel.starts_with(&format!("{base}/")))
    })
}

/// Shell-style glob (`*` and `?` match any chars, including `/`, as
/// Python's fnmatch does).
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

pub struct ClaimHook {
    pub task_id: String,
    pub sup: Weak<Supervisor>,
    pub workdir: PathBuf,
}

impl ToolHook for ClaimHook {
    fn before_tool(&self, tool: &dyn Tool, args: &Value) -> Result<(), String> {
        if !matches!(tool.name(), "edit" | "write") {
            return Ok(());
        }
        let Some(raw) = args
            .get("path")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            return Ok(());
        };
        let path = crate::tools::resolve_path(&self.workdir, raw);
        let root = self.workdir.canonicalize().unwrap_or(self.workdir.clone());
        let full = path
            .parent()
            .and_then(|p| p.canonicalize().ok())
            .map(|p| p.join(path.file_name().unwrap_or_default()))
            .unwrap_or(path);
        let Ok(rel) = full.strip_prefix(&root) else {
            return Ok(());
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        let Some(sup) = self.sup.upgrade() else {
            return Ok(());
        };
        match sup.claimed_by(&self.task_id, &rel) {
            Some((id, agent, title)) => Err(format!(
                "{rel} 由任务 {id}（{agent}：{title}）认领。不要直接改它：用 need(on=[\"{id}\"]) 等它完成，或用 ask_main 说明你为什么需要改。"
            )),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_patterns() {
        let p = vec!["src/db/**".to_string(), "*.md".to_string()];
        assert!(matches("src/db", &p));
        assert!(matches("src/db/x/y.rs", &p));
        assert!(matches("README.md", &p));
        assert!(!matches("src/api.rs", &p));
    }
}
