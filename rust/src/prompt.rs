//! System prompt assembly.
//!
//! The tools section comes from the registry, so the prompt always matches
//! the tools the model can call. Daily mode swaps the intro and guidelines
//! and adds the frozen memory snapshot; Code Mode adds the `run_code` SDK.
//! `AGENTS.md` files (project + global) and the skills catalog are appended
//! as context blocks.

use std::path::{Path, PathBuf};

use crate::config::limbo_home;
use crate::memory::MemorySnapshot;
use crate::skills::{discover_skills, format_skills};
use crate::tools::{PLAN_MODE_INSTRUCTION, ToolRegistry};

const BASE_INTRO: &str = "You are an expert coding assistant operating inside limbo, a coding agent harness. You help \
users by reading files, executing commands, editing code, and writing new files.\n\n";

const DAILY_INTRO: &str = "You are a capable personal assistant operating inside limbo. You help the user with \
everyday non-coding work: researching topics, reading web pages and documents, writing and editing text, organizing \
notes and files, and answering questions.\n\n";

const GUIDELINES_TAIL: &str = "- Use read to examine files before editing\n- Use edit for precise changes \
(edits[].old_text must match exactly)\n- Use write only for new files or complete rewrites\n- Be concise in your \
responses\n- Show file paths clearly when working with files";

const DAILY_GUIDELINES_TAIL: &str = "- For current facts, search (web_search) and read the sources (web_fetch) \
before answering; cite the URLs you relied on\n- Save substantial outputs (notes, reports, drafts) as files in the \
working directory and give their paths\n- Reply in the user's language; lead with the answer and keep it concise";

pub fn first_sentence(d: &str) -> &str {
    d.split(". ").next().unwrap_or(d).trim_end_matches('.')
}

/// Extra blocks the caller may append (project facts for delegation).
#[derive(Default)]
pub struct PromptExtras<'a> {
    pub memory: Option<&'a MemorySnapshot>,
    pub blocks: Vec<String>,
}

pub fn build_system_prompt(registry: &ToolRegistry, workdir: &Path) -> String {
    build_system_prompt_with(registry, workdir, &PromptExtras::default())
}

pub fn build_system_prompt_with(registry: &ToolRegistry, workdir: &Path, extras: &PromptExtras) -> String {
    let daily = registry.daily_mode;
    let mut s = String::from(if daily { DAILY_INTRO } else { BASE_INTRO });
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
    let has = |n: &str| registry.get(n).is_some();
    if has("bash") {
        s.push_str("\n- Prefer grep/find/ls tools over bash for file exploration");
        if has("write") {
            s.push_str("\n- Write multi-line scripts to a file with write, then run them with bash — avoid long heredoc one-liners");
        }
    }
    if daily && has("memory") {
        s.push_str("\n- When the user tells you something durable about themselves or how they like things done, save it with memory");
    }
    if daily && has("session_search") {
        s.push_str("\n- When the user refers to earlier conversations, look them up with session_search instead of asking them to repeat");
    }
    if daily && has("ask_user") {
        s.push_str("\n- When a decision is the user's to make, ask with ask_user (offer options) instead of guessing");
    }
    if daily && has("skill_manage") {
        s.push_str("\n- When you work out a reusable way to do a kind of task, or the user corrects how a task should be done, record it in a skill with skill_manage (patch the skill you used before creating a new one)");
    }
    if has("task") {
        s.push_str("\n- Delegate self-contained research or side work to a subagent with task; run several in parallel when they are independent");
    }
    s.push('\n');
    s.push_str(if daily { DAILY_GUIDELINES_TAIL } else { GUIDELINES_TAIL });
    if registry.code_mode {
        s.push_str("\n\n");
        s.push_str(crate::tools::run_code::CODE_ONLY_INSTRUCTION);
        s.push_str("\n\n");
        s.push_str(&crate::tools::run_code::render_tools_sdk(registry));
    }
    if registry.plan_mode {
        s.push_str("\n\n");
        s.push_str(PLAN_MODE_INSTRUCTION);
    }
    if daily
        && let Some(block) = extras.memory.and_then(|m| m.render())
    {
        s.push_str(&block);
    }
    s.push_str(&format!(
        "\n\nCurrent date: {}. Working directory: {}",
        chrono::Local::now().format("%Y-%m-%d"),
        workdir.display()
    ));
    for b in &extras.blocks {
        s.push_str(b);
    }
    if let Some(ctx) = context_block(workdir) {
        s.push_str(&ctx);
    }
    if has("read") || registry.code_mode {
        s.push_str(&format_skills(&discover_skills(workdir, daily)));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_lists_registered_tools_only() {
        let reg = ToolRegistry::core(false);
        let p = build_system_prompt(&reg, Path::new("/nonexistent"));
        assert!(p.contains("- read: Read the contents of a file"));
        assert!(!p.contains("- bash:"));
    }
}
