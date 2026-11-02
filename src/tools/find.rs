//! `find`: list files matching a glob. Uses the `ignore` crate (the walker
//! inside ripgrep), so — unlike the Python version — nested `.gitignore`
//! files and negations are honored too.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::truncate::{MAX_OUTPUT_BYTES, truncate_output};
use super::{Tool, ToolContext, arg_i64, arg_str, resolve_path};
use crate::models::ToolResult;

pub struct FindTool;

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &'static str {
        "find"
    }
    fn description(&self) -> String {
        "Find files by glob pattern. Respects .gitignore files (including nested ones). Patterns are matched \
         against paths relative to the search directory, e.g. '**/*.rs' or 'src/*.py'."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {"type": "string", "description": "Directory to search"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            "required": ["pattern"],
        })
    }
    fn replay_safe(&self) -> bool {
        true
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let pattern = arg_str(&args, "pattern").unwrap_or("").to_string();
        let raw = arg_str(&args, "path").unwrap_or(".").to_string();
        let limit = arg_i64(&args, "limit").unwrap_or(1000).max(1) as usize;
        let root = resolve_path(&ctx.workdir, &raw);
        let workdir = ctx.workdir.clone();
        tokio::task::spawn_blocking(move || find_files(&pattern, &raw, &root, &workdir, limit))
            .await
            .unwrap_or_else(|e| ToolResult::err(format!("find panicked: {e}")))
    }
}

fn find_files(
    pattern: &str,
    raw: &str,
    root: &std::path::Path,
    workdir: &std::path::Path,
    limit: usize,
) -> ToolResult {
    if !root.exists() {
        return ToolResult::err(format!("Path not found: {raw}"));
    }
    // A bare `*.rs` should match at any depth, the way people expect.
    let pat = if pattern.contains('/') {
        pattern.to_string()
    } else {
        format!("**/{pattern}")
    };
    let glob = match globset::GlobBuilder::new(&pat)
        .literal_separator(true)
        .build()
    {
        Ok(g) => g.compile_matcher(),
        Err(e) => return ToolResult::err(format!("Invalid glob pattern: {e}")),
    };
    let mut results = Vec::new();
    let walker = ignore::WalkBuilder::new(root)
        .hidden(false)
        .require_git(false)
        .filter_entry(|e| e.file_name() != ".git")
        .build();
    for entry in walker.flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        let rel = path.strip_prefix(root).unwrap_or(path);
        if !glob.is_match(rel) {
            continue;
        }
        results.push(
            path.strip_prefix(workdir)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| path.display().to_string()),
        );
        if results.len() >= limit {
            break;
        }
    }
    results.sort();
    if results.is_empty() {
        return ToolResult::ok("No matches.");
    }
    ToolResult::ok(truncate_output(results.join("\n"), MAX_OUTPUT_BYTES))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn respects_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        std::fs::create_dir_all(d.join("src/deep")).unwrap();
        std::fs::create_dir_all(d.join("target")).unwrap();
        std::fs::write(d.join(".gitignore"), "target/\n").unwrap();
        std::fs::write(d.join("src/a.rs"), "").unwrap();
        std::fs::write(d.join("src/deep/b.rs"), "").unwrap();
        std::fs::write(d.join("target/c.rs"), "").unwrap();
        let ctx = ToolContext::simple(d);
        let r = FindTool.run(json!({"pattern": "*.rs"}), &ctx).await;
        assert_eq!(r.output.unwrap(), "src/a.rs\nsrc/deep/b.rs");
    }
}
