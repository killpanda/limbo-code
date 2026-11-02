//! `edit`: exact-text replacements, all matched against the *original* file.
//!
//! Rules (same as the Python version): every `old_text` must occur exactly
//! once, edits must not overlap, a UTF-8 BOM is hidden from matching and
//! restored on write, and CRLF files keep their line endings.

use async_trait::async_trait;
use serde_json::{Value, json};

use super::{Tool, ToolContext, arg_str, mutation_lock, resolve_path};
use crate::models::ToolResult;

pub struct EditTool;

#[derive(Debug, Clone, PartialEq)]
pub struct Replacement {
    pub old_text: String,
    pub new_text: String,
}

/// Accept `edits: [...]`, `edits` as a JSON string, and the legacy
/// top-level `old_text`/`new_text` shape.
pub fn prepare_edits(args: &Value) -> Result<Vec<Replacement>, String> {
    let mut edits = args.get("edits").cloned();
    if let Some(Value::String(s)) = &edits {
        edits = serde_json::from_str(s).ok();
    }
    let mut out = Vec::new();
    if let Some(Value::Array(items)) = edits {
        for it in items {
            if let (Some(o), Some(n)) = (
                it.get("old_text").and_then(Value::as_str),
                it.get("new_text").and_then(Value::as_str),
            ) {
                out.push(Replacement {
                    old_text: o.into(),
                    new_text: n.into(),
                });
            }
        }
    }
    if let (Some(o), Some(n)) = (arg_str(args, "old_text"), arg_str(args, "new_text")) {
        out.push(Replacement {
            old_text: o.into(),
            new_text: n.into(),
        });
    }
    if out.is_empty() {
        return Err("edits must contain at least one {old_text, new_text} replacement.".into());
    }
    Ok(out)
}

/// Apply every edit against the same original content.
pub fn apply_edits(content: &str, edits: &[Replacement], path: &str) -> Result<String, String> {
    let single = edits.len() == 1;
    let mut matched: Vec<(usize, usize, usize, &str)> = Vec::new(); // (edit idx, start, len, new)
    for (i, e) in edits.iter().enumerate() {
        if e.old_text.is_empty() {
            return Err(if single {
                "old_text cannot be empty.".into()
            } else {
                format!("edits[{i}].old_text cannot be empty.")
            });
        }
        let count = content.matches(e.old_text.as_str()).count();
        if count == 0 {
            return Err(if single {
                format!("old_text not found in {path}.")
            } else {
                format!(
                    "Could not find edits[{i}] in {path}. The old_text must match exactly including all whitespace and newlines."
                )
            });
        }
        if count > 1 {
            return Err(if single {
                format!("Text must be unique in {path}.")
            } else {
                format!(
                    "Found {count} occurrences of edits[{i}] in {path}. Each old_text must be unique. Please provide more context to make it unique."
                )
            });
        }
        let start = content.find(e.old_text.as_str()).unwrap();
        matched.push((i, start, e.old_text.len(), e.new_text.as_str()));
    }
    matched.sort_by_key(|m| m.1);
    for w in matched.windows(2) {
        if w[0].1 + w[0].2 > w[1].1 {
            return Err(format!(
                "edits[{}] and edits[{}] overlap in {path}. Merge them into one edit or target disjoint regions.",
                w[0].0, w[1].0
            ));
        }
    }
    let mut out = content.to_string();
    for (_, start, len, new) in matched.iter().rev() {
        out.replace_range(*start..*start + *len, new);
    }
    if out == content {
        return Err(format!("No changes to {path}."));
    }
    Ok(out)
}

/// A unified diff for display (and for the model to double-check).
pub fn unified_diff(old: &str, new: &str) -> String {
    similar::TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(3)
        .header("", "")
        .to_string()
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }
    fn description(&self) -> String {
        "Edit a single file using exact text replacement. Every edits[].old_text must match a unique, \
         non-overlapping region of the original file. If two changes affect the same block or nearby lines, \
         merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions \
         just to connect distant changes. The displayed diff may not clearly indicate trailing-newline changes."
            .into()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {"type": "string", "description": "Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].old_text in the same call."},
                            "new_text": {"type": "string", "description": "Replacement text for this targeted edit."},
                        },
                        "required": ["old_text", "new_text"],
                    },
                },
            },
            "required": ["path", "edits"],
        })
    }

    async fn run(&self, args: Value, ctx: &ToolContext) -> ToolResult {
        let raw = arg_str(&args, "path").unwrap_or("").to_string();
        let edits = match prepare_edits(&args) {
            Ok(e) => e,
            Err(e) => return ToolResult::err(e),
        };
        let target = resolve_path(&ctx.workdir, &raw);
        if !target.exists() {
            return ToolResult::err(format!("File not found: {raw}"));
        }
        let lock = mutation_lock(&target);
        let _guard = lock.lock().await;
        let raw_content = match tokio::fs::read(&target).await {
            Ok(b) => match String::from_utf8(b) {
                Ok(s) => s,
                Err(e) => return ToolResult::err(format!("File is not UTF-8 text: {e}")),
            },
            Err(e) => return ToolResult::err(format!("Could not read file: {e}")),
        };
        let (bom, body) = match raw_content.strip_prefix('\u{feff}') {
            Some(rest) => ("\u{feff}", rest),
            None => ("", raw_content.as_str()),
        };
        let crlf = body
            .find('\n')
            .is_some_and(|i| i > 0 && body.as_bytes()[i - 1] == b'\r');
        let norm = |s: &str| s.replace("\r\n", "\n").replace('\r', "\n");
        let content = norm(body);
        let edits: Vec<Replacement> = edits
            .iter()
            .map(|e| Replacement {
                old_text: norm(&e.old_text),
                new_text: norm(&e.new_text),
            })
            .collect();
        let new_content = match apply_edits(&content, &edits, &raw) {
            Ok(c) => c,
            Err(e) => return ToolResult::err(e),
        };
        let diff = unified_diff(&content, &new_content);
        let to_write = if crlf {
            new_content.replace('\n', "\r\n")
        } else {
            new_content
        };
        if let Err(e) = tokio::fs::write(&target, format!("{bom}{to_write}")).await {
            return ToolResult::err(format!("Could not write file: {e}"));
        }
        let head = if edits.len() > 1 {
            format!("Edited {raw} ({} replacements).", edits.len())
        } else {
            format!("Edited {raw}.")
        };
        ToolResult::ok(format!("{head}\n{diff}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(o: &str, n: &str) -> Replacement {
        Replacement {
            old_text: o.into(),
            new_text: n.into(),
        }
    }

    #[test]
    fn applies_multiple_against_original() {
        let out = apply_edits("a b c", &[r("a", "x"), r("c", "z")], "f").unwrap();
        assert_eq!(out, "x b z");
    }

    #[test]
    fn rejects_ambiguous_and_overlapping() {
        assert!(
            apply_edits("aa", &[r("a", "b")], "f")
                .unwrap_err()
                .contains("unique")
        );
        assert!(
            apply_edits("abc", &[r("ab", "x"), r("bc", "y")], "f")
                .unwrap_err()
                .contains("overlap")
        );
        assert!(
            apply_edits("abc", &[r("zz", "y")], "f")
                .unwrap_err()
                .contains("not found")
        );
    }

    #[test]
    fn legacy_and_string_shapes() {
        let e = prepare_edits(&json!({"old_text": "a", "new_text": "b"})).unwrap();
        assert_eq!(e, vec![r("a", "b")]);
        let e =
            prepare_edits(&json!({"edits": "[{\"old_text\":\"a\",\"new_text\":\"b\"}]"})).unwrap();
        assert_eq!(e.len(), 1);
    }

    #[tokio::test]
    async fn keeps_crlf_and_bom() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.txt");
        std::fs::write(&p, "\u{feff}one\r\ntwo\r\n").unwrap();
        let ctx = ToolContext::simple(dir.path());
        let res = EditTool
            .run(
                json!({"path": "f.txt", "edits": [{"old_text": "one\ntwo", "new_text": "1\n2"}]}),
                &ctx,
            )
            .await;
        assert!(res.success, "{res:?}");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "\u{feff}1\r\n2\r\n");
    }
}
