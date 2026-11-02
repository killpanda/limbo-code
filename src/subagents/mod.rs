//! Background subagents and the task board.
//!
//! The main agent delegates with `plan_tasks`; each task runs as a child
//! agent in the background ([`supervisor`]), with a profile ([`profiles`])
//! deciding its tools and instructions, optional git worktree isolation
//! ([`worktree`]) and path claims ([`hooks`]). Results come back to the
//! main agent as `<subagent_updates>` messages.
//!
//! This module also holds the user-side verbs shared by the frontends:
//! `/task`, `/agents` and `@name` routing.

pub mod hooks;
pub mod profiles;
pub mod supervisor;
pub mod tools;
pub mod worktree;

use serde_json::Value;

pub use supervisor::Supervisor;

/// One-line summary of a tool call for progress display.
pub fn describe_call(name: &str, args: &Value) -> String {
    let key = match name {
        "read" | "write" | "edit" | "ls" => "path",
        "bash" => "command",
        "grep" | "find" => "pattern",
        "web_fetch" => "url",
        "web_search" | "session_search" => "query",
        "run_code" => "description",
        _ => "",
    };
    let detail = args.get(key).and_then(Value::as_str).unwrap_or("");
    let detail: String = detail
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(80)
        .collect();
    if detail.is_empty() {
        name.to_string()
    } else {
        format!("{name} {detail}")
    }
}

pub const TASK_USAGE: &str = "用法：/task 查看任务板 · /task <id> 查看详情 · \
/task <pause|resume|cancel|retry|approve|reject|merge> <id|all> [说明] · \
/task prio <id> <n> · /task edit <id> <新说明> · /task send <id> <消息>；也可以直接 @子代理名 <消息>";

const ACTIONS: [&str; 7] = [
    "pause", "resume", "cancel", "retry", "approve", "reject", "merge",
];

pub fn task_command(sup: &Supervisor, arg: &str) -> String {
    let parts: Vec<&str> = arg
        .splitn(3, char::is_whitespace)
        .filter(|p| !p.is_empty())
        .collect();
    let Some(first) = parts.first() else {
        return if sup.tasks().is_empty() {
            TASK_USAGE.into()
        } else {
            format!("{}\n\n{TASK_USAGE}", sup.status_text(None))
        };
    };
    let action = first.to_lowercase();
    if matches!(action.as_str(), "help" | "-h" | "--help") {
        return TASK_USAGE.into();
    }
    if parts.len() == 1 {
        return sup.status_text(Some(first));
    }
    let r = parts[1];
    let rest = parts.get(2).copied().unwrap_or("").trim();
    if ACTIONS.contains(&action.as_str()) {
        let fresh = action == "retry" && rest == "fresh";
        return sup.control(
            r,
            &action,
            if fresh { "" } else { rest },
            None,
            fresh,
            None,
            None,
            "user",
        );
    }
    match action.as_str() {
        "prio" | "priority" | "reprioritize" => match rest.parse::<i64>() {
            Ok(p) => sup.control(r, "reprioritize", "", Some(p), false, None, None, "user"),
            Err(_) => "用法：/task prio <id> <整数，越大越先>".into(),
        },
        "edit" if rest.is_empty() => "用法：/task edit <id> <新的任务说明>".into(),
        "edit" => sup.control(r, "edit", "", None, false, Some(rest.into()), None, "user"),
        "send" => sup.send(r, rest, "steer", "user"),
        _ => format!("未知动作 {action}\n{TASK_USAGE}"),
    }
}

/// `@name text` → the reply line; None when it names no subagent (the text
/// is then an ordinary message).
pub fn direct_message(sup: &Supervisor, text: &str) -> Option<String> {
    let t = text.trim();
    let rest = t.strip_prefix('@')?;
    let (name, msg) = rest.split_once(char::is_whitespace)?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '-')
        || msg.trim().is_empty()
    {
        return None;
    }
    sup.find(name)?;
    Some(sup.send(name, msg.trim(), "steer", "user"))
}

pub fn agents_command(sup: &Supervisor, arg: &str) -> String {
    match arg.trim().to_lowercase().as_str() {
        "scan" => {
            let mut lines = vec![
                profiles::format_facts(sup.facts()),
                String::new(),
                "可用的子代理角色：".into(),
            ];
            for p in sup.profiles.values() {
                let source = if p.builtin {
                    "内置".to_string()
                } else {
                    p.path
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                };
                lines.push(format!(
                    "- {}（{}）{} · {source}",
                    p.name, p.tools, p.description
                ));
            }
            let sug = profiles::suggest_profiles(sup.facts(), &sup.workdir);
            if !sug.is_empty() {
                lines.push(String::new());
                lines.push("建议的项目角色（/agents init 写入 .agents/agents/）：".into());
                lines.extend(sug.iter().map(|(n, _, _, r)| format!("- {n}：{r}")));
            }
            lines.join("\n")
        }
        "init" => {
            let sug = profiles::suggest_profiles(sup.facts(), &sup.workdir);
            if sug.is_empty() {
                return format!(
                    "没有新的建议角色（{}/ 已覆盖，或项目没有可切分的子包）",
                    profiles::PROJECT_AGENTS_DIR
                );
            }
            let mut written = vec![];
            for (_, path, content, reason) in sug {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if std::fs::write(&path, content).is_ok() {
                    let rel = path.strip_prefix(&sup.workdir).unwrap_or(&path);
                    written.push(format!("- {}（{reason}）", rel.display()));
                }
            }
            format!("已写入项目角色，下次会话生效：\n{}", written.join("\n"))
        }
        "" | "list" | "status" => sup.status_text(None),
        _ => "用法：/agents（面板）· /agents scan（项目画像与角色）· /agents init（生成项目角色）"
            .into(),
    }
}
