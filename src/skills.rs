//! Skills: directories with a `SKILL.md` (frontmatter `name`, `description`
//! + an instruction body), the same convention Claude Code and pi use.
//!
//! Discovery (later sources win on name collisions):
//! - bundled daily pack (daily sessions only, lowest priority)
//! - global:  `~/.agents/skills/<name>/SKILL.md` (shared with other agents;
//!   where `npx skills add -g` installs)
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

static BUNDLED: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/assets/daily_skills");

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

/// The cross-agent global skills dir (`~/.agents/skills`).
pub fn global_skills_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".agents").join("skills"))
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
///
/// Only top-level keys count (a `name:` nested under `metadata:` is not the
/// skill's name). Values may be plain, quoted (`"..."` / `'...'`), continued
/// on indented lines, or YAML block scalars (`>`, `|`, with chomping
/// indicators): the forms skill installers commonly write.
pub fn parse_skill_md(text: &str, fallback: &str) -> (String, String, String, bool) {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if let Some(rest) = text.strip_prefix("---")
        && let Some(end) = rest.find("\n---")
    {
        let header = &rest[..end];
        let body = &rest[end + 4..];
        // Drop the rest of the closing `---` line.
        let body = body.split_once('\n').map_or("", |(_, b)| b);
        let body = body.trim_start_matches(['\n', '\r']).to_string();
        let (mut name, mut desc, mut disable) = (fallback.to_string(), String::new(), false);
        for (k, v) in frontmatter_fields(header) {
            match k.as_str() {
                "name" if !v.is_empty() => name = v,
                "description" => desc = v,
                "disable-model-invocation" => {
                    disable = matches!(v.to_lowercase().as_str(), "true" | "yes" | "1")
                }
                _ => {}
            }
        }
        return (name, desc, body, disable);
    }
    (fallback.to_string(), String::new(), text.to_string(), false)
}

/// Top-level `key: value` pairs of a frontmatter block, values unquoted and
/// block scalars folded.
fn frontmatter_fields(header: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = header.lines().collect();
    let indented = |l: &str| l.starts_with([' ', '\t']);
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        i += 1;
        if indented(line) || line.trim_start().starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let key = k.trim().to_string();
        let v = v.trim();
        // The indented lines (and blank lines between them) that follow.
        let mut cont = Vec::new();
        while i < lines.len() && (indented(lines[i]) || lines[i].trim().is_empty()) {
            cont.push(lines[i]);
            i += 1;
        }
        while cont.last().is_some_and(|l| l.trim().is_empty()) {
            cont.pop();
        }
        let value = if let Some(ind) = v.strip_prefix(['>', '|']) {
            let literal = v.starts_with('|');
            let keep_newline = ind.contains('+');
            block_scalar(&cont, literal, keep_newline)
        } else if v.starts_with('"') || v.starts_with('\'') {
            let mut joined = v.to_string();
            for l in &cont {
                joined.push(' ');
                joined.push_str(l.trim());
            }
            unquote(&joined)
        } else {
            let mut joined = v.to_string();
            for l in &cont {
                let l = l.trim();
                if !l.is_empty() {
                    if !joined.is_empty() {
                        joined.push(' ');
                    }
                    joined.push_str(l);
                }
            }
            joined
        };
        out.push((key, value));
    }
    out
}

fn block_scalar(lines: &[&str], literal: bool, keep_newline: bool) -> String {
    let indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let body: Vec<&str> = lines
        .iter()
        .map(|l| l.get(indent..).unwrap_or("").trim_end())
        .collect();
    let mut s = if literal {
        body.join("\n")
    } else {
        // Folded: single newlines become spaces, blank lines stay newlines.
        let mut s = String::new();
        let mut prev_blank = true;
        for l in body {
            if l.is_empty() {
                s.push('\n');
                prev_blank = true;
            } else {
                if !prev_blank {
                    s.push(' ');
                }
                s.push_str(l);
                prev_blank = false;
            }
        }
        s
    };
    if keep_newline {
        s.push('\n');
    }
    s.trim().to_string()
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    if v.len() >= 2 && v.starts_with('\'') && v.ends_with('\'') {
        return v[1..v.len() - 1].replace("''", "'");
    }
    if v.len() >= 2 && v.starts_with('"') && v.ends_with('"') {
        let inner = &v[1..v.len() - 1];
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(o) => out.push(o),
                None => out.push('\\'),
            }
        }
        return out;
    }
    v.to_string()
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
        let (name, mut description, body, disable) = parse_skill_md(&text, &fallback);
        // Many published skills run past the spec's description limit; list
        // them with the description cut rather than hiding them.
        if description.chars().count() > MAX_DESCRIPTION_LENGTH {
            description = description
                .chars()
                .take(MAX_DESCRIPTION_LENGTH - 1)
                .collect();
            description.push('…');
        }
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
    let mut sources: Vec<PathBuf> = global_skills_dir().into_iter().collect();
    sources.push(user_skills_dir());
    sources.push(workdir.join(".agents/skills"));
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
    fn frontmatter_yaml_forms() {
        let folded = "---\nname: ste\ndescription: >\n  Report results\n  in short sentences.\nlicense: MIT\n---\nBody";
        let (n, d, b, _) = parse_skill_md(folded, "x");
        assert_eq!(
            (n.as_str(), d.as_str(), b.as_str()),
            ("ste", "Report results in short sentences.", "Body")
        );

        let literal = "---\nname: a\ndescription: |-\n  line one\n  line two\n---\n";
        assert_eq!(parse_skill_md(literal, "x").1, "line one\nline two");

        let quoted = "---\nname: lark-im\nversion: 1.0.0\ndescription: \"飞书: 收发消息 \\\"群聊\\\"\"\nmetadata:\n  name: nested\n  requires:\n    bins: [\"lark-cli\"]\n---\nBody";
        let (n, d, _, _) = parse_skill_md(quoted, "x");
        assert_eq!(
            (n.as_str(), d.as_str()),
            ("lark-im", "飞书: 收发消息 \"群聊\"")
        );

        let single = "---\nname: s\ndescription: 'it''s fine'\n---\n";
        assert_eq!(parse_skill_md(single, "x").1, "it's fine");

        let plain_multi = "---\nname: p\ndescription: first part\n  second part\n---\n";
        assert_eq!(parse_skill_md(plain_multi, "x").1, "first part second part");

        let crlf = "---\r\nname: c\r\ndescription: >\r\n  windows\r\n  file\r\n---\r\nBody";
        let (n, d, b, _) = parse_skill_md(crlf, "x");
        assert_eq!(
            (n.as_str(), d.as_str(), b.as_str()),
            ("c", "windows file", "Body")
        );
    }

    #[test]
    fn long_description_is_cut_not_skipped() {
        let dir = std::env::temp_dir().join(format!("limbo-skills-{}", std::process::id()));
        let skill = dir.join("long");
        std::fs::create_dir_all(&skill).unwrap();
        let desc = "x".repeat(MAX_DESCRIPTION_LENGTH + 60);
        std::fs::write(
            skill.join("SKILL.md"),
            format!("---\nname: long\ndescription: \"{desc}\"\n---\nBody"),
        )
        .unwrap();
        let found = load_from(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].description.chars().count(), MAX_DESCRIPTION_LENGTH);
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
