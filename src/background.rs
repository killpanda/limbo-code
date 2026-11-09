//! Events that happen outside a running turn (background subagents,
//! skill reviews). One process-wide channel: producers call [`emit`], the
//! frontend takes the receiver once at startup.
//!
//! A process that hosts several conversations at once (the chat gateway)
//! registers a **route** per main session: [`emit_for`] delivers an event
//! of that session to its route, and falls back to the shared channel when
//! nobody registered one (TUI / headless never do).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub enum BackgroundEvent {
    /// A line worth showing the user (`kind`: result, alert, question, info).
    Notice { kind: String, text: String },
    /// A task's status changed (board panels refresh).
    TaskChanged {
        id: String,
        status: String,
        title: String,
    },
    /// The model proposed a plan that waits for the user's approval.
    PlanProposed(Vec<String>),
    /// Notices were queued for an idle main agent: start a turn for them.
    MainWake,
    /// The TUI login dialog's key check finished (`error`: rejected).
    AuthChecked {
        provider: String,
        model: String,
        key: String,
        detail: String,
        error: bool,
    },
}

type Chan = (
    mpsc::UnboundedSender<BackgroundEvent>,
    Mutex<Option<mpsc::UnboundedReceiver<BackgroundEvent>>>,
);

fn chan() -> &'static Chan {
    static CHAN: OnceLock<Chan> = OnceLock::new();
    CHAN.get_or_init(|| {
        let (tx, rx) = mpsc::unbounded_channel();
        (tx, Mutex::new(Some(rx)))
    })
}

pub fn emit(ev: BackgroundEvent) {
    let _ = chan().0.send(ev);
}

type Routes = Mutex<HashMap<String, mpsc::UnboundedSender<BackgroundEvent>>>;

fn routes() -> &'static Routes {
    static ROUTES: OnceLock<Routes> = OnceLock::new();
    ROUTES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// An event that belongs to main session `session` (subagent notices).
pub fn emit_for(session: &str, ev: BackgroundEvent) {
    let routed = routes().lock().unwrap().get(session).cloned();
    match routed {
        Some(tx) if !tx.is_closed() => {
            let _ = tx.send(ev);
        }
        _ => emit(ev),
    }
}

/// Deliver `session`'s events to `tx` from now on (replacing an older route).
pub fn route(session: &str, tx: mpsc::UnboundedSender<BackgroundEvent>) {
    routes().lock().unwrap().insert(session.to_string(), tx);
}

/// Stop routing `session` (its events go to the shared channel again).
pub fn unroute(session: &str) {
    routes().lock().unwrap().remove(session);
}

/// The receiving end (the first caller gets it).
pub fn take_receiver() -> Option<mpsc::UnboundedReceiver<BackgroundEvent>> {
    chan().1.lock().unwrap().take()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_events_reach_their_session_only() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        route("route-test-a", tx);
        emit_for("route-test-a", BackgroundEvent::MainWake);
        assert!(matches!(rx.try_recv(), Ok(BackgroundEvent::MainWake)));
        unroute("route-test-a");
        emit_for("route-test-a", BackgroundEvent::MainWake);
        assert!(rx.try_recv().is_err());
    }
}
