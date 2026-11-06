//! Skill reviews (`/learn`) and when daily mode starts one by itself.
//!
//! The skill-reviewer is a fork subagent: every run re-reads the whole
//! conversation, so running it after every busy turn is expensive and
//! mostly ends in "Nothing to save". The trigger watches the main
//! conversation's events and only fires when all of these hold:
//!
//! - the turn finished cleanly (no interrupt, no error);
//! - enough tool *rounds* (responses that called tools; a parallel batch is
//!   one round) ran since the last skill update or review;
//! - at least `cooldown` turns passed since the last review;
//! - a signal suggests there is something to learn: the user corrected the
//!   assistant, a tool failed and the retry succeeded, or a skill was used.
//!   Without a signal, three times the threshold still fires as a fallback.
//!
//! A review that ends in "Nothing to save" doubles the threshold (capped
//! at `MAX_BACKOFF`×); one that changed a skill brings it back to the base.

use std::collections::HashSet;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::json;

use crate::agent::{Agent, AgentEvent};

pub const MAX_BACKOFF: u32 = 8;
pub const FALLBACK_FACTOR: u32 = 3;
pub const NOTHING_SAVED: [&str; 3] = ["nothing to save", "没有需要保存", "无需保存"];

fn correction_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)不要|别再|别用|不用再|太长|太短|太啰嗦|啰嗦|以后|下次|每次都|总是|记住|不对|错了|不是这样|\b(?:don'?t|do not|stop doing|always|never|next time|from now on|wrong)\b|\btoo (?:long|short|verbose|wordy)\b",
        )
        .unwrap()
    })
}

pub fn is_correction(text: &str) -> bool {
    static REMINDER: OnceLock<Regex> = OnceLock::new();
    let re =
        REMINDER.get_or_init(|| Regex::new(r"(?s)<system-reminder>.*?</system-reminder>").unwrap());
    correction_re().is_match(&re.replace_all(text, ""))
}

#[derive(Debug, Clone)]
pub struct LearnTrigger {
    pub every: u32,
    pub cooldown: u32,
    pub rounds: u32,
    pub turns_since_review: u32,
    pub backoff: u32,
    pub signals: HashSet<&'static str>,
    review_id: Option<String>,
    in_round: bool,
    failed: HashSet<String>,
    active: bool,
}

impl LearnTrigger {
    pub fn new(every: u32, cooldown: u32) -> Self {
        Self {
            every,
            cooldown,
            rounds: 0,
            // Start past the cooldown: the first review needs no wait.
            turns_since_review: cooldown,
            backoff: 1,
            signals: HashSet::new(),
            review_id: None,
            in_round: false,
            failed: HashSet::new(),
            active: false,
        }
    }

    pub fn threshold(&self) -> u32 {
        self.every * self.backoff
    }

    /// Feed one event of the main conversation's turn.
    pub fn observe(&mut self, ev: &AgentEvent) {
        if let AgentEvent::ToolCall { .. } = ev {
            self.active = true;
            if !self.in_round {
                self.rounds += 1;
            }
            self.in_round = true;
            return;
        }
        // Results of the current batch keep the round open.
        if !matches!(
            ev,
            AgentEvent::ToolResult { .. } | AgentEvent::ToolProgress { .. }
        ) {
            self.in_round = false;
        }
        if !matches!(ev, AgentEvent::Compaction { .. } | AgentEvent::Usage { .. }) {
            self.active = true;
        }
        match ev {
            AgentEvent::ToolResult {
                name,
                success,
                output,
                ..
            } => {
                if !success {
                    self.failed.insert(name.clone());
                    return;
                }
                if name == "skill_manage" {
                    // The main agent updated a skill itself.
                    self.rounds = 0;
                    self.signals.clear();
                } else if self.failed.contains(name) {
                    self.signals.insert("recovered");
                }
                if name == "read" && output.contains("SKILL.md") {
                    self.signals.insert("skill_used");
                }
            }
            AgentEvent::Steer(t) => self.observe_user_text(t),
            _ => {}
        }
    }

    /// `read` of a SKILL.md (from the call's arguments).
    pub fn observe_call(&mut self, name: &str, args: &serde_json::Value) {
        if name == "read"
            && args
                .get("path")
                .and_then(|p| p.as_str())
                .is_some_and(|p| p.ends_with("SKILL.md"))
        {
            self.signals.insert("skill_used");
        }
    }

    pub fn observe_user_text(&mut self, text: &str) {
        if is_correction(text) {
            self.signals.insert("correction");
        }
    }

    /// The turn ended in `state` ("done", "error", "interrupted"); true when
    /// a review should start.
    pub fn end_turn(&mut self, state: &str) -> bool {
        let active = std::mem::take(&mut self.active);
        self.in_round = false;
        self.failed.clear();
        if !active {
            return false;
        }
        self.turns_since_review += 1;
        if state != "done" || self.every == 0 {
            return false;
        }
        if self.turns_since_review < self.cooldown || self.rounds < self.threshold() {
            return false;
        }
        !self.signals.is_empty() || self.rounds >= self.threshold() * FALLBACK_FACTOR
    }

    pub fn review_started(&mut self, task_id: Option<String>) {
        self.rounds = 0;
        self.turns_since_review = 0;
        self.signals.clear();
        self.review_id = task_id;
    }

    /// Adjust the backoff once the last review has an outcome
    /// (`status`, `summary`).
    pub fn settle(&mut self, outcome: impl Fn(&str) -> Option<(String, String)>) {
        let Some(id) = self.review_id.clone() else {
            return;
        };
        let Some((status, summary)) = outcome(&id) else {
            return;
        };
        self.review_id = None;
        if status != "done" {
            return;
        }
        let s = summary.to_lowercase();
        if NOTHING_SAVED.iter().any(|p| s.contains(p)) {
            self.backoff = (self.backoff * 2).min(MAX_BACKOFF);
        } else {
            self.backoff = 1;
        }
    }
}

/// Start a skill-reviewer subagent over the conversation so far. Returns
/// what to tell the user (`auto` stays quiet when there is nothing to do).
pub fn start_review(agent: &mut Agent, auto: bool) -> String {
    let quiet = |m: &str| if auto { String::new() } else { m.to_string() };
    if !agent.daily_mode() {
        return quiet("/learn 只在 daily 模式下可用");
    }
    let Some(sup) = agent.supervisor.clone() else {
        return quiet("/learn 需要子代理（[subagents] enabled = true）");
    };
    if sup.review_running() {
        return quiet("已有一个复盘在进行中");
    }
    agent.sync_supervisor();
    let r = sup.plan(
        &[json!({
            "title": "复盘：把这次对话的经验写进 skill",
            "brief": "Review the conversation above and update the daily skills as your instructions say.",
            "profile": "skill-reviewer",
            "context": "fork",
        })],
        "user",
    );
    let Some(id) = r.ids.first().cloned() else {
        return quiet(&format!("复盘没能开始：{}", r.message));
    };
    agent.learn.lock().unwrap().review_started(Some(id.clone()));
    format!("已在后台开始复盘（{id}），有改动会告诉你")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call() -> AgentEvent {
        AgentEvent::ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: json!({}),
            provider: crate::capability::Provider::Tool,
        }
    }
    fn result(name: &str, ok: bool) -> AgentEvent {
        AgentEvent::ToolResult {
            id: "1".into(),
            name: name.into(),
            success: ok,
            output: String::new(),
        }
    }

    #[test]
    fn corrections() {
        assert!(is_correction("以后别用表格"));
        assert!(is_correction("That's too long"));
        assert!(!is_correction(
            "<system-reminder>always</system-reminder> thanks"
        ));
    }

    #[test]
    fn fires_after_rounds_with_signal_and_backs_off() {
        let mut t = LearnTrigger::new(2, 0);
        for _ in 0..2 {
            t.observe(&call());
            t.observe(&result("bash", false));
            t.observe(&AgentEvent::Text("x".into()));
            t.observe(&call());
            t.observe(&result("bash", true));
            t.observe(&AgentEvent::Text("y".into()));
        }
        assert!(t.signals.contains("recovered"));
        assert!(t.end_turn("done"));
        t.review_started(Some("t1".into()));
        t.settle(|_| Some(("done".into(), "Nothing to save.".into())));
        assert_eq!(t.backoff, 2);
        assert_eq!(t.threshold(), 4);
    }

    #[test]
    fn no_signal_needs_fallback() {
        let mut t = LearnTrigger::new(1, 0);
        t.observe(&call());
        t.observe(&AgentEvent::Text("a".into()));
        assert!(!t.end_turn("done"));
        for _ in 0..3 {
            t.observe(&call());
            t.observe(&AgentEvent::Text("a".into()));
        }
        assert!(t.end_turn("done"));
    }
}
