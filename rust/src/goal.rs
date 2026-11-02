//! Closed-loop goal mode (`/goal`): state, prompt templates, verify runner.
//!
//! The loop's gate is the verify command's exit code: 0 means the goal is
//! met; anything else feeds the failure output back verbatim into the next
//! round. The orchestration (turn → verify → next round) lives in
//! [`crate::pump`].
//!
//! 🦀 Rust notes
//! - [`GoalCommand`] is an enum with data in some variants; `match` on it
//!   forces every caller to handle all three cases.
//! - [`GoalState`] derives `Serialize`/`Deserialize` so it rides inside
//!   the session meta exactly like the Python pydantic model does.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

use crate::tools::truncate::{DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail};

pub const STATUS_ACTIVE: &str = "active";
pub const STATUS_PASSED: &str = "passed";
pub const STATUS_EXHAUSTED: &str = "exhausted";
pub const STATUS_CLEARED: &str = "cleared";

/// One session-scoped goal (persisted on `SessionMeta.goal`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalState {
    pub text: String,
    /// Shell command whose exit code gates the loop; empty = no auto-verify.
    #[serde(default)]
    pub verify_command: String,
    #[serde(default)]
    pub rounds_completed: u32,
    #[serde(default = "default_rounds")]
    pub max_rounds: u32,
    #[serde(default = "default_status")]
    pub status: String,
}

fn default_rounds() -> u32 {
    10
}
fn default_status() -> String {
    STATUS_ACTIVE.into()
}

impl GoalState {
    pub fn new(text: &str, verify_command: &str, max_rounds: u32) -> Self {
        Self {
            text: text.into(),
            verify_command: verify_command.into(),
            rounds_completed: 0,
            max_rounds,
            status: STATUS_ACTIVE.into(),
        }
    }
    pub fn is_live(&self) -> bool {
        self.status != STATUS_CLEARED
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GoalCommand {
    Set(String),
    Clear,
    Query,
}

/// `/goal <目标>` · `/goal clear` · `/goal` (query).
pub fn parse_goal_args(arg: &str) -> GoalCommand {
    match arg.trim() {
        "" => GoalCommand::Query,
        "clear" => GoalCommand::Clear,
        t => GoalCommand::Set(t.to_string()),
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VerifyResult {
    /// None when the command never produced an exit code (timeout/cancel).
    pub exit_code: Option<i32>,
    pub output_tail: String,
    pub timed_out: bool,
    pub cancelled: bool,
}

/// Run the acceptance command (`bash -c`, own process group). Output is
/// merged and tail-truncated — failure stacks live at the end.
pub async fn run_verify(
    command: &str,
    workdir: &Path,
    timeout_ms: u64,
    cancel: &CancellationToken,
) -> VerifyResult {
    let mut cmd = tokio::process::Command::new("bash");
    cmd.arg("-c")
        .arg(format!("exec 2>&1; {command}"))
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .process_group(0)
        .kill_on_drop(true);
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return VerifyResult {
                exit_code: Some(127),
                output_tail: format!("could not start bash: {e}"),
                ..Default::default()
            };
        }
    };
    let pid = child.id();
    let mut stdout = child.stdout.take().expect("piped stdout");
    let reader = tokio::spawn(async move {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf).await;
        buf
    });
    let kill = |pid: Option<u32>| {
        if let Some(pid) = pid {
            // SAFETY: killpg on our own child's process group.
            unsafe {
                libc::killpg(pid as i32, libc::SIGKILL);
            }
        }
    };
    tokio::select! {
        status = child.wait() => {
            let buf = reader.await.unwrap_or_default();
            let text = String::from_utf8_lossy(&buf);
            VerifyResult {
                exit_code: Some(status.ok().and_then(|s| s.code()).unwrap_or(-1)),
                output_tail: truncate_tail(&text, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES).content,
                ..Default::default()
            }
        }
        _ = tokio::time::sleep(Duration::from_millis(timeout_ms)) => {
            kill(pid);
            VerifyResult { timed_out: true, ..Default::default() }
        }
        _ = cancel.cancelled() => {
            kill(pid);
            VerifyResult { cancelled: true, ..Default::default() }
        }
    }
}

// -- prompt templates ----------------------------------------------------------

pub const VERIFY_PROPOSAL_TAG: &str = "verify_proposal";
const MAX_PROPOSALS: usize = 3;

/// First (proposal) round: explore and propose acceptance commands, then stop.
pub fn build_proposer_prompt(state: &GoalState) -> String {
    format!(
        "🎯 Goal 模式已启动\n\n目标：{text}\n\n你的第一个任务（先不要动手修改任何代码）：探索当前仓库，找出能客观判定\
该目标是否完成的验收命令（例如跑哪些测试、lint、typecheck）。要求：\n\
- 命令必须能在当前工作目录直接用 bash 运行，退出码 0 = 通过、非 0 = 未通过；\n\
- 优先复用仓库已有的测试/检查入口（Makefile、pyproject、package.json 等）；\n\
- 给出 1–{MAX_PROPOSALS} 条候选（可以是一条组合命令）；\n\
- 若该目标没有客观可判定的验收方式（例如纯审美类目标），如实说明，不要硬造。\n\
探索完成后，用以下格式输出提议，然后停止，等我确认：\n\
<{tag}>\n<command>命令1</command>\n<command>命令2（可选）</command>\n<rationale>一句话说明理由</rationale>\n</{tag}>\n\
若没有客观验收方式，输出 <{tag}><none/></{tag}> 并解释。",
        text = state.text,
        tag = VERIFY_PROPOSAL_TAG,
    )
}

/// `None` = no proposal block; `Some([])` = explicit `<none/>`.
pub fn parse_verify_proposal(text: &str) -> Option<Vec<String>> {
    let block_re = Regex::new(&format!(
        r"(?s)<{VERIFY_PROPOSAL_TAG}>(.*?)</{VERIFY_PROPOSAL_TAG}>"
    ))
    .unwrap();
    let block = block_re.captures_iter(text).last()?.get(1)?.as_str();
    if block.contains("<none/>") || block.contains("<none>") {
        return Some(Vec::new());
    }
    let cmd_re = Regex::new(r"(?s)<command>(.*?)</command>").unwrap();
    let mut out: Vec<String> = Vec::new();
    for c in cmd_re.captures_iter(block) {
        let command = c[1].split_whitespace().collect::<Vec<_>>().join(" ");
        if !command.is_empty() && !out.contains(&command) {
            out.push(command);
        }
    }
    out.truncate(MAX_PROPOSALS);
    Some(out)
}

pub fn build_initial_prompt(state: &GoalState) -> String {
    let gate = if state.verify_command.is_empty() {
        "尚未设置验收命令；本轮结束后不会自动验收续轮。".to_string()
    } else {
        format!(
            "验收标准将由命令 `{}` 判定（退出码 0 = 通过）。",
            state.verify_command
        )
    };
    format!(
        "🎯 Goal 模式已启动\n\n目标：{}\n\n{gate}\n\n循环规则：\n\
- 请开始朝着目标工作；你认为完成后，我会执行验收命令。\n\
- 验收未通过时，失败输出会原样交回给你，继续修复，直到通过。\n\
- 若验收标准本身有歧义，先向我提问，不要自行猜测。",
        state.text
    )
}

/// Self-prompt for the next round: failure output fed back verbatim.
pub fn build_goal_prompt(state: &GoalState, result: &VerifyResult) -> String {
    let round = state.rounds_completed + 1;
    let outcome = if result.timed_out {
        format!("验收命令 `{}` 执行超时，未完成。", state.verify_command)
    } else {
        format!(
            "验收命令 `{}` 退出码 {}，失败输出如下（原样）：\n<verify_output>\n{}\n</verify_output>",
            state.verify_command,
            result.exit_code.unwrap_or(-1),
            result.output_tail
        )
    };
    format!(
        "🎯 Goal（第 {round} 轮验收未通过，进入第 {} 轮）\n\n目标：{}\n\n{outcome}\n\n循环规则：\n\
- 阅读失败输出，定位并修复，然后继续朝验收标准推进。\n\
- 验收标准全部满足才算完成；不要凭印象猜测哪里出了问题。\n\
- 若验收标准本身有歧义，先向用户提问，不要自行猜测。",
        round + 1,
        state.text
    )
}

/// Budget exhausted: summarize without tools.
pub fn build_wrapup_prompt(state: &GoalState) -> String {
    format!(
        "🎯 Goal 已达最大轮次（{} 轮）仍未通过验收。\n\n目标：{}\n\n请不要使用任何工具，直接总结汇报：\n\
1. 已完成什么；\n2. 卡在哪里（最近失败的原因）；\n3. 建议怎么继续。\n\
最后告诉用户：发送任意消息（如“继续”）即可带着新预算继续闭环；/goal clear 可退出 goal 模式。",
        state.max_rounds, state.text
    )
}

/// Advance the goal after one verify (one verify = one round).
pub fn next_state(state: &GoalState, result: &VerifyResult) -> GoalState {
    let mut s = state.clone();
    s.rounds_completed += 1;
    s.status = if result.exit_code == Some(0) {
        STATUS_PASSED
    } else if s.rounds_completed >= s.max_rounds {
        STATUS_EXHAUSTED
    } else {
        STATUS_ACTIVE
    }
    .into();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_args() {
        assert_eq!(parse_goal_args(""), GoalCommand::Query);
        assert_eq!(parse_goal_args(" clear "), GoalCommand::Clear);
        assert_eq!(
            parse_goal_args("make tests pass"),
            GoalCommand::Set("make tests pass".into())
        );
    }

    #[test]
    fn parses_proposals() {
        let t = "x <verify_proposal><command>cargo  test</command><command>cargo test</command>\
                 <command>make lint</command></verify_proposal>";
        assert_eq!(
            parse_verify_proposal(t),
            Some(vec!["cargo test".to_string(), "make lint".to_string()])
        );
        assert_eq!(
            parse_verify_proposal("<verify_proposal><none/></verify_proposal>"),
            Some(vec![])
        );
        assert_eq!(parse_verify_proposal("nothing"), None);
    }

    #[test]
    fn state_machine() {
        let s = GoalState::new("g", "false", 2);
        let fail = VerifyResult {
            exit_code: Some(1),
            ..Default::default()
        };
        let s1 = next_state(&s, &fail);
        assert_eq!(s1.status, STATUS_ACTIVE);
        let s2 = next_state(&s1, &fail);
        assert_eq!(s2.status, STATUS_EXHAUSTED);
        let ok = VerifyResult {
            exit_code: Some(0),
            ..Default::default()
        };
        assert_eq!(next_state(&s, &ok).status, STATUS_PASSED);
    }

    #[tokio::test]
    async fn verify_runs_and_cancels() {
        let dir = tempfile::tempdir().unwrap();
        let cancel = CancellationToken::new();
        let r = run_verify("echo hi; exit 3", dir.path(), 5000, &cancel).await;
        assert_eq!(r.exit_code, Some(3));
        assert!(r.output_tail.contains("hi"));
        let r = run_verify("sleep 5", dir.path(), 100, &cancel).await;
        assert!(r.timed_out);
    }
}
