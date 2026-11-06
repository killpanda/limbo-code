//! The turn pump: runs one *task* — a user turn plus everything that
//! follows from it — on an agent the frontend handed over.
//!
//! - **follow-ups**: messages queued with Alt+Enter wait for the whole task
//!   and then become the next turn (steers are injected inside the turn by
//!   the agent itself);
//! - **the goal closed loop**: at a quiet turn boundary, run the verify
//!   command and feed failures back verbatim into the next round, until it
//!   passes or the round budget is spent;
//! - **`[queue] after_failed_run`**: after a failed turn, "keep"/"withdraw"
//!   stop the task instead of feeding queued messages one by one into a
//!   provider that keeps failing.
//!
//! Every round is a complete `Agent::run` turn, so per-round iteration
//! limits, auto-compaction and tracing behave exactly as for a typed turn.

use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentEvent, EventTx, FOLLOW_UP, STEER};
use crate::effect::Effect;
use crate::goal::{
    self, GoalState, STATUS_ACTIVE, STATUS_CLEARED, STATUS_EXHAUSTED, STATUS_PASSED,
    build_goal_prompt, build_wrapup_prompt, next_state, run_verify,
};
use crate::journal::{ActionKind, ActionPhase, ActionRecord};
use crate::models::Attachment;
use crate::posture::{Posture, Strategy};
use crate::session::QueuedMessageRecord;

/// Run one task to completion (or interruption).
///
/// Around the rounds: the task board learns the main agent is busy (its
/// notices are steered in instead of waking it) and gets undelivered
/// notices flushed ahead of the user's message; daily mode's skill-review
/// trigger watches every event and may start a review afterwards.
pub async fn run_task(
    agent: &mut Agent,
    input: String,
    attachments: Vec<Attachment>,
    tx: &EventTx,
    cancel: &CancellationToken,
) {
    if let Some(sup) = &agent.supervisor {
        sup.set_main_busy(true);
        sup.flush_notices();
    }
    agent.sync_skills();
    agent.learn.lock().unwrap().observe_user_text(&input);
    // Tap the event stream for the learn trigger.
    let (itx, mut irx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let learn = agent.learn.clone();
    let out = tx.clone();
    let tap = tokio::spawn(async move {
        while let Some(ev) = irx.recv().await {
            {
                let mut l = learn.lock().unwrap();
                if let AgentEvent::ToolCall {
                    name, arguments, ..
                } = &ev
                {
                    l.observe_call(name, arguments);
                }
                l.observe(&ev);
            }
            let _ = out.send(ev);
        }
    });
    run_rounds(agent, input, attachments, &itx, cancel).await;
    drop(itx);
    let _ = tap.await;
    let state = if agent.was_interrupted || cancel.is_cancelled() {
        "interrupted"
    } else if agent.last_run_failed {
        "error"
    } else {
        "done"
    };
    let sup = agent.supervisor.clone();
    if let Some(sup) = &sup {
        agent
            .learn
            .lock()
            .unwrap()
            .settle(|id| sup.outcome_of(id).map(|o| (o.status, o.summary)));
    }
    let review = agent.daily_mode() && agent.learn.lock().unwrap().end_turn(state);
    if review {
        let msg = crate::learn::start_review(agent, true);
        if !msg.is_empty() {
            let _ = tx.send(AgentEvent::Info(msg));
        }
    }
    if let Some(sup) = &sup {
        sup.set_main_busy(false);
    }
}

async fn run_rounds(
    agent: &mut Agent,
    input: String,
    attachments: Vec<Attachment>,
    tx: &EventTx,
    cancel: &CancellationToken,
) {
    let mut next = (input, attachments);
    loop {
        run_once(agent, next.0, next.1, tx, cancel).await;
        if agent.was_interrupted || cancel.is_cancelled() {
            return;
        }
        if holds_queue(agent) {
            return;
        }
        let items = agent.steer.drain_modes(&[STEER, FOLLOW_UP]);
        let Some(batch) = join(agent, items, tx) else {
            return;
        };
        next = batch;
    }
}

fn holds_queue(agent: &Agent) -> bool {
    agent.last_run_failed
        && matches!(
            agent.config.queue.after_failed_run.as_str(),
            "keep" | "withdraw"
        )
}

/// Join drained items into one turn's input; announce each to the frontend.
fn join(
    agent: &Agent,
    items: Vec<QueuedMessageRecord>,
    tx: &EventTx,
) -> Option<(String, Vec<Attachment>)> {
    if items.is_empty() {
        return None;
    }
    let ids: Vec<String> = items.iter().map(|i| i.id.clone()).collect();
    for i in &items {
        if i.source == "user" {
            let _ = tx.send(AgentEvent::Steer(i.text.clone()));
        }
    }
    // The next turn's user message carries them; record that they left the queue.
    agent.steer.journal_dequeued(&ids);
    let text = items
        .iter()
        .map(|i| i.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let atts = items.into_iter().flat_map(|i| i.attachments).collect();
    Some((text, atts))
}

fn save(agent: &mut Agent, state: GoalState) {
    agent.save_goal(Some(state));
}

async fn run_once(
    agent: &mut Agent,
    input: String,
    attachments: Vec<Attachment>,
    tx: &EventTx,
    cancel: &CancellationToken,
) {
    // Any user message reactivates an exhausted goal with a fresh budget.
    if let Some(mut state) = agent.meta.goal.clone()
        && state.status == STATUS_EXHAUSTED
    {
        state.status = STATUS_ACTIVE.into();
        state.rounds_completed = 0;
        let max = state.max_rounds;
        save(agent, state);
        agent.trace.log("goal_resumed", json!({"max_rounds": max}));
        let _ = tx.send(AgentEvent::GoalResumed { max_rounds: max });
    }
    let mut next = (input, attachments);
    loop {
        agent.run(next.0, next.1, tx, cancel).await;
        if agent.was_interrupted || cancel.is_cancelled() {
            agent.trace.log("goal_interrupted", json!({}));
            return;
        }
        if holds_queue(agent) && !agent.steer.is_empty() {
            let policy = agent.config.queue.after_failed_run.clone();
            let mut items = agent.steer.items();
            if policy == "withdraw" {
                items.retain(|i| agent.steer.cancel(&i.id).is_some());
            }
            agent.trace.log(
                "queue_held",
                json!({"policy": policy, "ids": items.iter().map(|i| i.id.clone()).collect::<Vec<_>>()}),
            );
            let _ = tx.send(AgentEvent::QueueHeld {
                policy,
                texts: items
                    .into_iter()
                    .filter(|i| i.source == "user")
                    .map(|i| i.text)
                    .collect(),
            });
            return;
        }
        // Leftover steers (error exits) outrank goal continuation.
        let steers = agent.steer.drain_modes(&[STEER]);
        if let Some(batch) = join(agent, steers, tx) {
            next = batch;
            continue;
        }
        // Only the goal strategy continues past a quiet turn boundary.
        if Posture::strategy(&agent.meta) != Strategy::Goal {
            return;
        }
        let Some(state) = agent.meta.goal.clone() else {
            return;
        };
        if state.status != STATUS_ACTIVE || state.verify_command.is_empty() || agent.ended_by_tool {
            return;
        }
        let round = state.rounds_completed + 1;
        let _ = tx.send(AgentEvent::GoalVerifyStarted {
            command: state.verify_command.clone(),
            round,
            max_rounds: state.max_rounds,
        });
        let action_id = format!("verify-{round}");
        let verify_action = |phase| {
            ActionRecord::new(ActionKind::Verify, &action_id, &state.verify_command, phase)
                .with_effect(Some(Effect::Exec))
        };
        agent.journal_action(&verify_action(ActionPhase::Started));
        let result = run_verify(
            &state.verify_command,
            &agent.workdir,
            agent.config.goal.verify_timeout_ms,
            cancel,
        )
        .await;
        let finished = match (result.exit_code, result.cancelled, result.timed_out) {
            (Some(0), ..) => verify_action(ActionPhase::Completed).with_detail("exit 0"),
            (_, true, _) => verify_action(ActionPhase::Failed).with_detail("cancelled"),
            (_, _, true) => verify_action(ActionPhase::Failed).with_detail("timed out"),
            (code, ..) => verify_action(ActionPhase::Failed).with_detail(format!(
                "exit {}",
                code.map_or("?".into(), |c| c.to_string())
            )),
        };
        agent.journal_action(&finished);
        agent.trace.log(
            "goal_verify",
            json!({"command": state.verify_command, "exit_code": result.exit_code, "round": round,
                   "timed_out": result.timed_out, "cancelled": result.cancelled}),
        );
        let _ = tx.send(AgentEvent::GoalVerifyResult {
            exit_code: result.exit_code,
            output_tail: result.output_tail.clone(),
            timed_out: result.timed_out,
            cancelled: result.cancelled,
            round,
            max_rounds: state.max_rounds,
        });
        if result.cancelled {
            return;
        }
        let state = next_state(&state, &result);
        save(agent, state.clone());
        if state.status == STATUS_PASSED {
            agent
                .trace
                .log("goal_complete", json!({"rounds": state.rounds_completed}));
            let _ = tx.send(AgentEvent::GoalPassed {
                rounds: state.rounds_completed,
            });
            return;
        }
        if state.status == STATUS_EXHAUSTED {
            agent
                .trace
                .log("goal_exhausted", json!({"rounds": state.rounds_completed}));
            agent
                .run(build_wrapup_prompt(&state), Vec::new(), tx, cancel)
                .await;
            let _ = tx.send(AgentEvent::GoalExhausted {
                rounds: state.rounds_completed,
            });
            return;
        }
        agent.trace.log(
            "goal_round",
            json!({"next_round": state.rounds_completed + 1}),
        );
        next = (build_goal_prompt(&state, &result), Vec::new());
    }
}

// -- /goal verbs (the frontends call these between tasks) ---------------------

/// Set a new goal (no verify yet); returns the proposal-round prompt.
pub fn set_goal(agent: &mut Agent, text: &str) -> String {
    let state = GoalState::new(text, "", agent.config.goal.max_rounds);
    agent.trace.log(
        "goal_set",
        json!({"text": text, "verify_command": "", "max_rounds": state.max_rounds}),
    );
    let prompt = goal::build_proposer_prompt(&state);
    save(agent, state);
    prompt
}

/// Adopt a verify command (fresh gate = fresh budget); returns the first
/// round's prompt, or None when the goal went away meanwhile.
pub fn set_verify(agent: &mut Agent, command: &str) -> Option<String> {
    let mut state = agent.meta.goal.clone()?;
    if state.status == STATUS_CLEARED {
        return None;
    }
    state.verify_command = command.into();
    state.rounds_completed = 0;
    state.status = STATUS_ACTIVE.into();
    let prompt = goal::build_initial_prompt(&state);
    save(agent, state);
    Some(prompt)
}

/// Exit goal mode; false when there was no live goal.
pub fn clear_goal(agent: &mut Agent) -> bool {
    match agent.meta.goal.clone() {
        Some(mut s) if s.status != STATUS_CLEARED => {
            s.status = STATUS_CLEARED.into();
            save(agent, s);
            agent.trace.log("goal_cleared", json!({}));
            true
        }
        _ => false,
    }
}

/// The live goal (cleared ones are history).
pub fn live_goal(agent: &Agent) -> Option<&GoalState> {
    agent.meta.goal.as_ref().filter(|g| g.is_live())
}

/// After the proposal round: the model's last answer, parsed.
pub fn take_proposal(agent: &Agent) -> Option<Vec<String>> {
    let text = agent
        .messages
        .iter()
        .rev()
        .find(|m| m.role == crate::models::Role::Assistant && !m.text().is_empty())
        .map(|m| m.text().to_string())
        .unwrap_or_default();
    goal::parse_verify_proposal(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tests::{fake_server, sse, test_config};

    fn text_reply(t: &str) -> String {
        sse(&[
            json!({"choices": [{"delta": {"content": t}}]}),
            json!({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ])
    }

    #[tokio::test]
    async fn goal_loop_feeds_failures_back_until_pass() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("ok");
        let (base, requests) = fake_server(vec![
            text_reply("working"),
            text_reply("fixed"),
            text_reply("done"),
        ])
        .await;
        let mut agent = Agent::new(test_config(&base), dir.path(), dir.path()).unwrap();
        set_goal(&mut agent, "make the marker");
        // round 1 fails (no marker); the follow-up round creates it
        let cmd = format!("test -f {}", marker.display());
        let prompt = set_verify(&mut agent, &cmd).unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let m2 = marker.clone();
        // create the marker after the first verify failed
        let watcher = tokio::spawn(async move {
            while let Some(ev) = rx.recv().await {
                if let AgentEvent::GoalVerifyResult {
                    exit_code: Some(1), ..
                } = ev
                {
                    std::fs::write(&m2, "").unwrap();
                }
                if let AgentEvent::GoalPassed { rounds } = ev {
                    return rounds;
                }
            }
            0
        });
        run_task(&mut agent, prompt, vec![], &tx, &cancel).await;
        drop(tx);
        assert_eq!(watcher.await.unwrap(), 2);
        assert_eq!(agent.meta.goal.as_ref().unwrap().status, STATUS_PASSED);
        let reqs = requests.lock().unwrap();
        let second = reqs[1]["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(second.contains("第 1 轮验收未通过"), "{second}");
    }

    #[tokio::test]
    async fn follow_ups_run_after_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let (base, requests) = fake_server(vec![text_reply("one"), text_reply("two")]).await;
        let mut agent = Agent::new(test_config(&base), dir.path(), dir.path()).unwrap();
        agent
            .steer
            .push_item("later please".into(), vec![], FOLLOW_UP, "user", vec![]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        run_task(
            &mut agent,
            "first".into(),
            vec![],
            &tx,
            &CancellationToken::new(),
        )
        .await;
        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 2);
        let last = reqs[1]["messages"].as_array().unwrap().last().unwrap()["content"].clone();
        assert_eq!(last, "later please");
    }
}
