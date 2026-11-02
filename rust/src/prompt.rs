//! System prompt assembly and skill discovery.
//!
//! The tools section comes from the registry, so the prompt always matches
//! the tools the model can call. `AGENTS.md` files (project + global) and
//! the skills catalog are appended as context blocks.

use std::path::{Path, PathBuf};

use crate::config::limbo_home;
use crate::tools::{PLAN_MODE_INSTRUCTION, ToolRegistry};

const BASE_INTRO: &str = "You are an expert coding assistant operating inside limbo, a coding agent harness. You help \
users by reading files, executing commands, editing code, and writing new files.\n\n";

const GUIDELINES_TAIL: &str = "- Use read to examine files before editing\n- Use edit for precise changes \
(edits[].old_text must match exactly)\n- Use write only for new files or complete rewrites\n- Be concise in your \
responses\n- Show file paths clearly when working with files";

fn first_sentence(d: &str) -> &str {
    d.split(". ").next().unwrap_or(d).trim_end_matches('.')
}

pub fn build_system_prompt(registry: &ToolRegistry, workdir: &Path) -> String {
    let mut s = String::from(BASE_INTRO);
    s.push_str("Available tools:");
    for def in registry.definitions() {
        let f = &def["function"];
        let desc = f["description"].as_str().unwrap_or("");
        s.push_str(&format!(
            "\n- {}: {}",
            f["name"].as_str().unwrap_or(""),
            first_sentence(desc)
        ));
    }
    s.push_str("\n\nGuidelines:");
    let names = registry.names();
    if names.contains(&"bash") {
        s.push_str("\n- Prefer grep/find/ls tools over bash for file exploration");
        if names.contains(&"write") {
            s.push_str("\n- Write multi-line scripts to a file with write, then run them with bash — avoid long heredoc one-liners");
        }
    }
    if names.contains(&"task") {
        s.push_str("\n- Delegate self-contained research or side work to a subagent with task; run several in parallel when they are independent");
    }
    s.push('\n');
    s.push_str(GUIDELINES_TAIL);
    if registry.plan_mode {
        s.push_str("\n\n");
        s.push_str(PLAN_MODE_INSTRUCTION);
    }
    s.push_str(&format!(
        "\n\nCurrent date: {}. Working directory: {}",
        chrono::Local::now().format("%Y-%m-%d"),
        workdir.display()
    ));
    if let Some(ctx) = context_block(workdir) {
        s.push_str(&ctx);
    }
    if names.contains(&"read") {
        s.push_str(&format_skills(&discover_skills(workdir)));
    }
    s
}

fn context_block(workdir: &Path) -> Option<String> {
    let files: Vec<(PathBuf, String)> = [workdir.join("AGENTS.md"), limbo_home().join("AGENTS.md")]
        .into_iter()
        .filter_map(|p| std::fs::read_to_string(&p).ok().map(|c| (p, c)))
        .collect();
    if files.is_empty() {
        return None;
    }
    let mut b =
        String::from("\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n");
    for (p, c) in files {
        b.push_str(&format!(
            "<project_instructions path=\"{}\">\n\n{c}\n\n</project_instructions>\n\n",
            p.display()
        ));
    }
    b.push_str("</project_context>\n");
    Some(b)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
    pub disable_model_invocation: bool,
}

/// Parse `---` frontmatter (`name`, `description`, `disable-model-invocation`).
pub fn parse_skill_md(text: &str, fallback: &str) -> (String, String, String, bool) {
    if let Some(rest) = text.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        let header = &rest[..end];
        let body = rest[end + 4..].trim_start_matches('\n').to_string();
        let (mut name, mut desc, mut disable) = (fallback.to_string(), String::new(), false);
        for line in header.lines() {
            if let Some((k, v)) = line.split_once(':') {
                let v = v.trim().trim_matches('"').to_string();
                match k.trim() {
                    "name" if !v.is_empty() => name = v,
                    "description" => desc = v,
                    "disable-model-invocation" => {
                        disable = matches!(v.to_lowercase().as_str(), "true" | "yes" | "1")
                    }
                    _ => {}
                }
            }
        }
        return (name, desc, body, disable);
    }
    (fallback.to_string(), String::new(), text.to_string(), false)
}

/// `~/.limbo/skills/*/SKILL.md` then `<workdir>/.agents/skills/*/SKILL.md`
/// (project wins on name collisions).
pub fn discover_skills(workdir: &Path) -> Vec<Skill> {
    let mut out: Vec<Skill> = Vec::new();
    for dir in [limbo_home().join("skills"), workdir.join(".agents/skills")] {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for e in entries {
            let p = e.path().join("SKILL.md");
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let fallback = e.file_name().to_string_lossy().into_owned();
            let (name, description, body, disable) = parse_skill_md(&text, &fallback);
            out.retain(|s| s.name != name);
            out.push(Skill {
                name,
                description,
                body,
                path: p,
                disable_model_invocation: disable,
            });
        }
    }
    out
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

pub fn format_skills(skills: &[Skill]) -> String {
    let visible: Vec<&Skill> = skills
        .iter()
        .filter(|s| !s.disable_model_invocation)
        .collect();
    if visible.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        String::new(),
        String::new(),
        "The following skills provide specialized instructions for specific tasks. When a task matches a skill's \
         description, use the read tool to load the skill's file. When a skill file references a relative path, \
         resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute \
         path in tool commands."
            .to_string(),
        String::new(),
        "<available_skills>".into(),
    ];
    for s in visible {
        lines.push("  <skill>".into());
        lines.push(format!("    <name>{}</name>", xml_escape(&s.name)));
        lines.push(format!(
            "    <description>{}</description>",
            xml_escape(&s.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            xml_escape(&s.path.display().to_string())
        ));
        lines.push("  </skill>".into());
    }
    lines.push("</available_skills>".into());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter() {
        let (n, d, b, dis) =
            parse_skill_md("---\nname: deploy\ndescription: Ship it\n---\nDo X", "x");
        assert_eq!(
            (n.as_str(), d.as_str(), b.as_str(), dis),
            ("deploy", "Ship it", "Do X", false)
        );
    }

    #[test]
    fn prompt_lists_registered_tools_only() {
        let reg = ToolRegistry::core(false);
        let p = build_system_prompt(&reg, Path::new("/nonexistent"));
        assert!(p.contains("- read: Read the contents of a file"));
        assert!(!p.contains("- bash:"));
    }
}
