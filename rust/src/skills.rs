//! Skills: directories with a `SKILL.md` (frontmatter `name`, `description`
//! + an instruction body), the same convention Claude Code and pi use.
//!
//! Discovery (later sources win on name collisions):
//! - bundled daily pack (daily sessions only, lowest priority)
//! - user:    `~/.limbo/skills/<name>/SKILL.md`
//! - project: `<workdir>/.agents/skills/<name>/SKILL.md`
//! - daily:   `~/.limbo/daily/skills/<name>/SKILL.md` (daily sessions only)
//!
//! The bundled pack ships *inside the binary* (`include_dir!`) and is
//! unpacked to `~/.limbo/daily/bundled/<version>/` the first time a daily
//! session needs it, so the SKILL.md paths the model reads are real files
//! and the helper scripts can run.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use include_dir::{Dir, include_dir};

use crate::config::limbo_home;

static BUNDLED: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../src/limbo/daily_skills");

pub const MAX_NAME_LENGTH: usize = 64;
pub const MAX_DESCRIPTION_LENGTH: usize = 1024;

#[derive(Debug, Clone, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
    pub disable_model_invocation: bool,
}

pub fn user_skills_dir() -> PathBuf {
    limbo_home().join("skills")
}
pub fn daily_skills_dir() -> PathBuf {
    limbo_home().join("daily").join("skills")
}

/// Where the bundled daily pack lives on disk (unpacked on first use).
pub fn bundled_daily_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = limbo_home()
            .join("daily")
            .join("bundled")
            .join(env!("CARGO_PKG_VERSION"));
        if !dir.join(".complete").exists() {
            let _ = std::fs::create_dir_all(&dir);
            if BUNDLED.extract(&dir).is_ok() {
                mark_scripts_executable(&dir);
                let _ = std::fs::write(dir.join(".complete"), "");
            }
        }
        dir
    })
    .clone()
}

fn mark_scripts_executable(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    for entry in ignore::WalkBuilder::new(dir)
        .hidden(false)
        .build()
        .flatten()
    {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == "py" || e == "sh")
            && let Ok(meta) = std::fs::metadata(p)
        {
            let mut perm = meta.permissions();
            perm.set_mode(perm.mode() | 0o755);
            let _ = std::fs::set_permissions(p, perm);
        }
    }
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
                let v = v.trim().to_string();
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

/// Validate per the Agent Skills spec; returns error messages.
pub fn validate_skill(name: &str, description: &str) -> Vec<String> {
    let mut errors = Vec::new();
    if name.chars().count() > MAX_NAME_LENGTH {
        errors.push(format!("name exceeds {MAX_NAME_LENGTH} characters"));
    }
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        errors.push("name must be lowercase a-z, 0-9 and hyphens only".into());
    } else if name.starts_with('-') || name.ends_with('-') || name.contains("--") {
        errors.push("name must not start/end with a hyphen or contain '--'".into());
    }
    if description.trim().is_empty() {
        errors.push("description is required".into());
    } else if description.chars().count() > MAX_DESCRIPTION_LENGTH {
        errors.push(format!(
            "description exceeds {MAX_DESCRIPTION_LENGTH} characters"
        ));
    }
    errors
}

fn load_from(dir: &Path) -> Vec<Skill> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    let mut out = Vec::new();
    for e in entries {
        let p = e.path().join("SKILL.md");
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        let fallback = e.file_name().to_string_lossy().into_owned();
        let (name, description, body, disable) = parse_skill_md(&text, &fallback);
        if !validate_skill(&name, &description).is_empty() {
            continue;
        }
        out.push(Skill {
            name,
            description,
            body,
            path: p,
            disable_model_invocation: disable,
        });
    }
    out
}

/// All skills visible to a session, sorted by name.
pub fn discover_skills(workdir: &Path, daily: bool) -> Vec<Skill> {
    let mut sources = vec![user_skills_dir(), workdir.join(".agents/skills")];
    if daily {
        sources.insert(0, bundled_daily_dir());
        sources.push(daily_skills_dir());
    }
    let mut by_name: std::collections::BTreeMap<String, Skill> = Default::default();
    for dir in sources {
        for s in load_from(&dir) {
            by_name.insert(s.name.clone(), s);
        }
    }
    by_name.into_values().collect()
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// The `<available_skills>` catalog (name/description/location only).
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
    fn validation() {
        assert!(validate_skill("weekly-report", "x").is_empty());
        assert!(!validate_skill("Bad_Name", "x").is_empty());
        assert!(!validate_skill("a--b", "x").is_empty());
        assert!(!validate_skill("ok", " ").is_empty());
    }

    #[test]
    fn bundled_pack_is_embedded() {
        assert!(BUNDLED.get_file("pdf/SKILL.md").is_some());
        assert!(BUNDLED.get_file("docx/SKILL.md").is_some());
    }
}
