//! Events that happen outside a running turn (background subagents,
//! skill reviews). One process-wide channel: producers call [`emit`], the
//! frontend takes the receiver once at startup.

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

/// The receiving end (the first caller gets it).
pub fn take_receiver() -> Option<mpsc::UnboundedReceiver<BackgroundEvent>> {
    chan().1.lock().unwrap().take()
}
