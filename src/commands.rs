//! Frontend-neutral pieces of the slash commands, shared by the TUI and the
//! headless REPL: toggle parsing, user-facing notices, `/btw` side queries.

use tokio::sync::mpsc;

use crate::llm::LlmClient;
use crate::models::{LlmEvent, Message};

// -- on/off arguments (/plan, /daily-mode, /code-mode) -----------------------

/// `on`/`off` (and friends) → that state, empty → flip `current`, else None.
pub fn parse_toggle(arg: &str, current: bool) -> Option<bool> {
    match arg.trim().to_lowercase().as_str() {
        "on" | "1" | "true" | "yes" | "开" | "开启" => Some(true),
        "off" | "0" | "false" | "no" | "关" | "关闭" => Some(false),
        "" => Some(!current),
        _ => None,
    }
}

// -- daily mode notices ------------------------------------------------------

pub const DAILY_USAGE: &str = "用法：/daily-mode on|off（无参则切换）";
pub const DAILY_DIVIDER_ON: &str =
    "☀ Daily mode · 通用助理口吻 · 读网页、长期记忆、翻历史会话已就绪";
pub const DAILY_DIVIDER_OFF: &str = "⌨ 回到 coding · 网页和记忆工具已收起";

pub fn daily_toggle_notice(enabled: bool) -> &'static str {
    if enabled {
        "Daily mode 已开启：通用助理口吻，新增读网页、长期记忆、翻历史会话等工具；/memory 查看记忆，/learn 把经验写进 skill"
    } else {
        "Daily mode 已关闭：回到 coding 模式"
    }
}

/// Why `web_search` is missing in daily mode, or None when it is registered
/// or the user turned it off (`[web_search] enabled = false`).
///
/// The tool is left out quietly when no DeepSeek key resolves, and a daily
/// session without search falls back to guessing or to `web_fetch` on URLs
/// it already knows, so say it once where the user can see it.
pub fn web_search_missing_notice(agent: &crate::agent::Agent) -> Option<String> {
    let s = &agent.config.web_search;
    if !agent.daily_mode() || !s.enabled || agent.registry.get("web_search").is_some() {
        return None;
    }
    Some(if s.provider != "deepseek" {
        format!(
            "web_search 未启用：不支持的 provider \"{}\"（目前只支持 deepseek）",
            s.provider
        )
    } else {
        "web_search 未启用：找不到 DeepSeek API key。在 ~/.limbo/config.toml 的 [web_search] 里设 \
         api_key，或 `limbo auth` 登录 deepseek，或设置 DEEPSEEK_API_KEY；不需要搜索可设 \
         [web_search] enabled = false"
            .to_string()
    })
}

pub fn daily_only_notice(command: &str) -> String {
    format!("{command} 只在 daily 模式可用（/daily-mode on 开启）")
}

pub fn unchanged_notice(what: &str, enabled: bool) -> String {
    format!("{what}已是{}状态", if enabled { "开启" } else { "关闭" })
}

// -- code mode / plan mode notices --------------------------------------------

pub const CODE_MODE_USAGE: &str = "用法：/code-mode on|off（无参则切换）";
pub const PLAN_USAGE: &str = "用法：/plan on|off（无参则切换）";

pub fn code_mode_notice(enabled: bool) -> &'static str {
    if enabled {
        "Code mode 已开启：模型改为写一段 Python 程序批量调用工具（run_code），中间结果不进上下文"
    } else {
        "Code mode 已关闭：恢复逐个调用工具"
    }
}

pub fn plan_notice(enabled: bool) -> &'static str {
    if enabled {
        "计划模式已开启：只读探索，edit/write 和会改动状态的命令会被拦下"
    } else {
        "计划模式已关闭：工具恢复正常"
    }
}

// -- /goal -------------------------------------------------------------------

pub fn goal_status_text(g: &crate::goal::GoalState) -> String {
    format!(
        "目标：{}\n验收命令：{}\n进度：{}/{} 轮 · 状态 {}",
        g.text,
        if g.verify_command.is_empty() {
            "（未设置）"
        } else {
            &g.verify_command
        },
        g.rounds_completed,
        g.max_rounds,
        g.status
    )
}

pub const NO_VERIFY_GATE: &str =
    "模型认为该目标没有客观可判定的验收方式，保持单轮模式；/goal clear 退出";
pub const VERIFY_SKIPPED: &str = "已跳过自动验收（单轮模式）；/goal clear 可退出 goal";

/// One line per goal event (headless output and TUI notices share it).
pub fn goal_event_line(ev: &crate::agent::AgentEvent) -> Option<String> {
    use crate::agent::AgentEvent::*;
    Some(match ev {
        GoalVerifyStarted {
            command,
            round,
            max_rounds,
        } => format!("🔍 第 {round}/{max_rounds} 轮验收：`{command}` · runtime"),
        GoalVerifyResult {
            exit_code,
            cancelled,
            timed_out,
            round,
            ..
        } => {
            if *cancelled {
                "验收已取消，闭环暂停（goal 保持 active）".into()
            } else if *timed_out {
                "✗ 验收命令执行超时，将带入下一轮处理".into()
            } else if *exit_code == Some(0) {
                "✅ 验收命令退出码 0".into()
            } else {
                format!(
                    "❌ 第 {round} 轮验收未通过（退出码 {}），失败输出将原样注入下一轮",
                    exit_code.map_or("?".to_string(), |c| c.to_string())
                )
            }
        }
        GoalPassed { rounds } => format!("✅ 目标达成（共 {rounds} 轮验收）"),
        GoalExhausted { rounds } => {
            format!("⚠️ 已达最大轮次（{rounds} 轮）：发送任意消息带新预算继续，/goal clear 退出")
        }
        GoalResumed { max_rounds } => format!("🎯 恢复 goal 闭环，预算已重置（{max_rounds} 轮）"),
        QueueHeld { policy, texts } => {
            if policy == "withdraw" {
                format!(
                    "上一轮出错，已撤回 {} 条排队消息：{}",
                    texts.len(),
                    texts
                        .iter()
                        .map(|t| t.lines().next().unwrap_or(""))
                        .collect::<Vec<_>>()
                        .join("；")
                )
            } else {
                format!(
                    "上一轮出错，{} 条排队消息保留，下次发送时一起处理",
                    texts.len()
                )
            }
        }
        _ => return None,
    })
}

// -- /btw ----------------------------------------------------------------------

/// The side-question instruction rides in the trailing user message (not
/// the system prompt) so the request prefix stays byte-identical to the main
/// conversation and provider prefix caches keep hitting.
pub const BTW_INSTRUCTION: &str = "【侧问模式】你现在是一个轻量级的“侧问”实例：用户正在主会话中进行一个更大的任务，这只是任务中途的一个快速问题。\n\
规则：\n\
1. 基于对话上下文直接、简洁地回答，一次性答完。\n\
2. 你没有任何工具可用，也不要承诺采取任何行动（不要说“我来试试”“让我查一下”）。\n\
3. 你的回答不会进入主会话历史，仅临时展示给用户。";

pub fn build_btw_messages(history: &[Message], question: &str) -> Vec<Message> {
    let mut messages = history.to_vec();
    messages.push(Message::user(format!(
        "{BTW_INSTRUCTION}\n\n## 用户的侧问\n\n{question}"
    )));
    messages
}

/// Stream a side answer: `Ok(text)` chunks, then one `Err` on failure.
pub fn btw_query(
    client: &LlmClient,
    history: &[Message],
    question: &str,
) -> mpsc::UnboundedReceiver<Result<String, String>> {
    let (tx, rx) = mpsc::unbounded_channel();
    let mut stream = client.chat(build_btw_messages(history, question), vec![]);
    tokio::spawn(async move {
        while let Some(item) = stream.recv().await {
            match item {
                Ok(LlmEvent::Text(t)) => {
                    if tx.send(Ok(t)).is_err() {
                        return;
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    let _ = tx.send(Err(e.to_string()));
                    return;
                }
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggles() {
        assert_eq!(parse_toggle("开启", false), Some(true));
        assert_eq!(parse_toggle(" OFF ", true), Some(false));
        assert_eq!(parse_toggle("", true), Some(false));
        assert_eq!(parse_toggle("maybe", true), None);
    }

    #[test]
    fn btw_keeps_history_prefix() {
        let h = vec![Message::system("sys"), Message::user("hi")];
        let m = build_btw_messages(&h, "what?");
        assert_eq!(m.len(), 3);
        assert!(m[2].text().contains("## 用户的侧问\n\nwhat?"));
    }
}
