//! `skill_manage` (daily mode): the agent curates its own skills.
//!
//! Every write lands in the daily skills dir (`~/.limbo/daily/skills`): a
//! bundled, user or project skill being changed is first copied there
//! (copy-on-write), and the daily copy then wins by name — so the agent
//! never edits the binary's pack, the user's coding skills or a repo.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use super::{Tool, ToolContext};
use crate::models::ToolResult;
use crate::skills::{Skill, daily_skills_dir, discover_skills, parse_skill_md, validate_skill};

const SUPPORT_DIRS: [&str; 3] = ["references", "templates", "scripts"];
const ACTIONS: [&str; 3] = ["create", "patch", "write_file"];

const DESCRIPTION: &str = "Create or improve skills: your procedural memory for recurring kinds of task (how to do X \
for this user). One call takes an operations array and applies it all-or-nothing. Ops: create (a whole new SKILL.md \
with frontmatter `name` and `description`), patch (replace old_string with new_string in a skill's SKILL.md, or in \
file_path; read the file first), write_file (add or replace a support file under references/, templates/ or \
scripts/). Skills are class-level (e.g. `weekly-report`, never `fix-bug-123`). Write lessons, not logs: an \
imperative rule plus why, one rule per bullet, no dates or session narration. The description is the trigger: say \
when to use the skill in its first sentence. Skills cannot be deleted with this tool.";

pub struct SkillManageTool;

#[derive(Default)]
struct Plan {
    staged: BTreeMap<PathBuf, String>,
    copies: BTreeMap<String, PathBuf>,
    created: Vec<String>,
}

fn require<'a>(op: &'a Map<String, Value>, key: &str) -> Result<&'a str, String> {
    match op.get(key).and_then(Value::as_str) {
        Some(v) if !v.is_empty() => Ok(v),
        _ => Err(format!(
            "{} needs {key}",
            op.get("action").and_then(Value::as_str).unwrap_or("")
        )),
    }
}

fn relative(raw: &str, allow_skill_md: bool) -> Result<PathBuf, String> {
    let rel = PathBuf::from(raw);
    if raw == "SKILL.md" && allow_skill_md {
        return Ok(rel);
    }
    let parts: Vec<Component> = rel.components().collect();
    let bad = rel.is_absolute()
        || parts.iter().any(|c| matches!(c, Component::ParentDir))
        || parts.len() < 2;
    if bad {
        return Err(format!(
            "file_path must be relative and start with one of: {}/",
            SUPPORT_DIRS.join(", ")
        ));
    }
    let first = parts[0].as_os_str().to_string_lossy();
    if !SUPPORT_DIRS.contains(&first.as_ref()) {
        return Err(format!(
            "file_path must start with one of: {}/",
            SUPPORT_DIRS.join(", ")
        ));
    }
    Ok(rel)
}

fn check_skill_md(text: &str, name: &str) -> Result<(), String> {
    let (parsed, description, _, _) = parse_skill_md(text, "");
    if !text.starts_with("---") || parsed.is_empty() {
        return Err("SKILL.md must start with --- frontmatter holding name and description".into());
    }
    if parsed != name {
        return Err(format!(
            "frontmatter name {parsed:?} does not match {name:?}"
        ));
    }
    let errors = validate_skill(&parsed, &description);
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }
    Ok(())
}

fn plan_one(
    op: &Value,
    root: &Path,
    skills: &BTreeMap<String, Skill>,
    plan: &mut Plan,
) -> Result<(), String> {
    let Some(op) = op.as_object() else {
        return Err("each operation must be an object".into());
    };
    let action = op.get("action").and_then(Value::as_str).unwrap_or("");
    if !ACTIONS.contains(&action) {
        return Err(format!("action must be one of: {}", ACTIONS.join(", ")));
    }
    let name = op.get("name").and_then(Value::as_str).unwrap_or("");
    if !validate_skill(name, "x").is_empty() {
        return Err(format!(
            "invalid skill name {name:?} (lowercase letters, digits, hyphens)"
        ));
    }
    let skill_dir = root.join(name);
    if action == "create" {
        if skills.contains_key(name) || plan.created.iter().any(|c| c == name) {
            return Err(format!("skill {name:?} already exists; patch it instead"));
        }
        let content = require(op, "content")?;
        check_skill_md(content, name)?;
        plan.staged
            .insert(skill_dir.join("SKILL.md"), content.into());
        plan.created.push(name.into());
        return Ok(());
    }
    let existing = skills.get(name);
    if existing.is_none() && !plan.created.iter().any(|c| c == name) {
        return Err(format!("no skill named {name:?}; create it first"));
    }
    if let Some(e) = existing
        && e.path.parent() != Some(skill_dir.as_path())
    {
        plan.copies
            .entry(name.into())
            .or_insert_with(|| e.path.parent().unwrap_or(root).to_path_buf());
    }
    let raw = op
        .get("file_path")
        .and_then(Value::as_str)
        .unwrap_or("SKILL.md");
    let rel = relative(raw, action == "patch")?;
    let target = skill_dir.join(&rel);
    if action == "write_file" {
        plan.staged.insert(target, require(op, "content")?.into());
        return Ok(());
    }
    let current = if let Some(s) = plan.staged.get(&target) {
        s.clone()
    } else {
        let source = if target.exists() {
            target.clone()
        } else if let Some(e) = existing {
            e.path.parent().unwrap_or(root).join(&rel)
        } else {
            target.clone()
        };
        std::fs::read_to_string(&source)
            .map_err(|e| format!("cannot read {} of {name:?}: {e}", rel.display()))?
    };
    let old = require(op, "old_string")?;
    let new = match op.get("new_string") {
        None | Some(Value::Null) => "",
        Some(Value::String(s)) => s.as_str(),
        _ => return Err("new_string must be a string".into()),
    };
    let count = current.matches(old).count();
    if count != 1 {
        return Err(format!(
            "old_string occurs {count} times in {} of {name:?}; it must occur once",
            rel.display()
        ));
    }
    let updated = current.replacen(old, new, 1);
    if rel == Path::new("SKILL.md") {
        check_skill_md(&updated, name)?;
    }
    plan.staged.insert(target, updated);
    Ok(())
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}

fn write_file(path: &Path, text: &str) -> std::io::Result<()> {
    if path.exists() {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let _ = std::fs::copy(path, path.with_file_name(format!("{name}.bak")));
    }
    crate::session::atomic_write(path, text.as_bytes())
}

#[async_trait]
impl Tool for SkillManageTool {
    fn name(&self) -> &'static str {
        "skill_manage"
    }
    fn effect(&self) -> crate::effect::Effect {
        crate::effect::Effect::Mutate
    }
    fn description(&self) -> String {
        DESCRIPTION.into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ACTIONS},
                            "name": {"type": "string", "description": "The skill's name."},
                            "content": {"type": "string", "description": "create: the full SKILL.md. write_file: the file."},
                            "old_string": {"type": "string", "description": "patch: exact text to replace (must occur once)."},
                            "new_string": {"type": "string", "description": "patch: the replacement (empty deletes)."},
                            "file_path": {"type": "string", "description": "write_file / patch: path relative to the skill dir, e.g. references/style.md (patch defaults to SKILL.md)."},
                        },
                        "required": ["action", "name"],
                    },
                },
            },
            "required": ["operations"],
        })
    }
    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let Some(ops) = args
            .get("operations")
            .and_then(Value::as_array)
            .filter(|o| !o.is_empty())
        else {
            return ToolResult::err("operations must be a non-empty list");
        };
        let root = daily_skills_dir();
        let skills: BTreeMap<String, Skill> = discover_skills(&ctx.workdir, true)
            .into_iter()
            .map(|s| (s.name.clone(), s))
            .collect();
        let mut plan = Plan::default();
        for (i, op) in ops.iter().enumerate() {
            if let Err(e) = plan_one(op, &root, &skills, &mut plan) {
                let w = if ops.len() > 1 {
                    format!("operation {}: ", i + 1)
                } else {
                    String::new()
                };
                return ToolResult::err(format!("{w}{e}"));
            }
        }
        for (name, source) in &plan.copies {
            let target = root.join(name);
            if !target.exists()
                && let Err(e) = copy_dir(source, &target)
            {
                return ToolResult::err(format!("could not copy {name}: {e}"));
            }
        }
        for (path, text) in &plan.staged {
            if let Err(e) = write_file(path, text) {
                return ToolResult::err(format!("could not write {}: {e}", path.display()));
            }
        }
        let touched: std::collections::BTreeSet<String> = plan
            .staged
            .keys()
            .filter_map(|p| p.strip_prefix(&root).ok())
            .filter_map(|r| r.components().next())
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let mut lines = vec![format!(
            "Updated skills: {} (in {}).",
            touched.into_iter().collect::<Vec<_>>().join(", "),
            root.display()
        )];
        let copied: Vec<&String> = plan
            .copies
            .keys()
            .filter(|n| !plan.created.contains(n))
            .collect();
        if !copied.is_empty() {
            lines.push(format!(
                "Copied {} into the daily skills dir first; the copy now replaces the original by name.",
                copied.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        lines.push("Changes apply the next time a skill is loaded.".into());
        ToolResult::ok(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_paths() {
        assert!(relative("../x", false).is_err());
        assert!(relative("notes/x.md", false).is_err());
        assert!(relative("references/x.md", false).is_ok());
        assert!(relative("SKILL.md", true).is_ok());
    }

    #[test]
    fn plans_create_then_patch() {
        let root = tempfile::tempdir().unwrap();
        let mut plan = Plan::default();
        let skills = BTreeMap::new();
        plan_one(
            &json!({"action": "create", "name": "weekly-report", "content": "---\nname: weekly-report\ndescription: Use for weekly reports.\n---\nBe brief."}),
            root.path(),
            &skills,
            &mut plan,
        )
        .unwrap();
        plan_one(
            &json!({"action": "patch", "name": "weekly-report", "old_string": "Be brief.", "new_string": "Be very brief."}),
            root.path(),
            &skills,
            &mut plan,
        )
        .unwrap();
        let text = plan.staged.values().next().unwrap();
        assert!(text.ends_with("Be very brief."));
        let err = plan_one(
            &json!({"action": "create", "name": "x", "content": "---\nname: y\ndescription: d\n---\n"}),
            root.path(),
            &skills,
            &mut plan,
        )
        .unwrap_err();
        assert!(err.contains("does not match"));
    }
}
