//! Lifecycle reports to terminal multiplexers / agent orchestrators.
//!
//! - **Herdr**: inside a Herdr pane (`HERDR_ENV=1`), state and session
//!   identity go to `herdr pane report-agent …` (fire-and-forget).
//! - **OSC 7501** program status: one root record written to the terminal
//!   so terminals that implement it can show working/blocked/done badges.
//!
//! Every reporter is a no-op unless its environment is detected, and none
//! may block or fail the session. Call sites only touch [`Reporters`].

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Mutex;

use base64::Engine;

/// Semantic lifecycle states (`done`/`error` are how a turn ended).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Working,
    Idle,
    Blocked,
    Done,
    Error,
}

impl State {
    fn as_str(self) -> &'static str {
        match self {
            State::Working => "working",
            State::Idle => "idle",
            State::Blocked => "blocked",
            State::Done => "done",
            State::Error => "error",
        }
    }
}

// ---------------------------------------------------------------------------
// Herdr

const HERDR_SOURCE: &str = "custom:limbo";
const AGENT: &str = "limbo";

struct Herdr {
    pane: String,
    bin: String,
    /// Starts at the wall clock: Herdr keeps a per-(pane, source)
    /// watermark across processes and drops anything at or below it.
    seq: u128,
}

impl Herdr {
    fn from_env(env: &dyn Fn(&str) -> Option<String>) -> Option<Self> {
        if env("HERDR_ENV").as_deref() != Some("1") {
            return None;
        }
        let pane = env("HERDR_PANE_ID").filter(|p| !p.is_empty())?;
        let bin = env("HERDR_BIN_PATH")
            .filter(|b| !b.is_empty())
            .or_else(|| which("herdr"))?;
        let seq = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Some(Self { pane, bin, seq })
    }

    fn next(&mut self) -> String {
        self.seq += 1;
        self.seq.to_string()
    }

    fn base(&mut self, verb: &str) -> Vec<String> {
        let seq = self.next();
        [
            &self.bin,
            "pane",
            verb,
            &self.pane,
            "--source",
            HERDR_SOURCE,
            "--agent",
            AGENT,
            "--seq",
            &seq,
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn report(&mut self, state: State, message: Option<&str>) {
        // Herdr only knows working/idle/blocked: a finished turn is idle.
        let st = match state {
            State::Done | State::Error => "idle",
            s => s.as_str(),
        };
        let mut cmd = self.base("report-agent");
        cmd.extend(["--state".into(), st.into()]);
        if let Some(m) = message.filter(|m| !m.is_empty()) {
            cmd.extend(["--message".into(), m.into()]);
        }
        spawn(&cmd);
    }

    fn report_session(&mut self, id: &str, path: &Path) {
        let mut cmd = self.base("report-agent-session");
        cmd.extend([
            "--agent-session-id".into(),
            id.into(),
            "--agent-session-path".into(),
            path.display().to_string(),
        ]);
        spawn(&cmd);
    }

    fn release(&mut self) {
        let cmd = self.base("release-agent");
        spawn(&cmd);
    }
}

fn which(bin: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(bin))
        .find(|p| p.is_file())
        .map(|p| p.display().to_string())
}

fn spawn(cmd: &[String]) {
    let _ = Command::new(&cmd[0])
        .args(&cmd[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

// ---------------------------------------------------------------------------
// OSC 7501

const MSG_MAX_CHARS: usize = 200;

pub fn encode_report(pairs: &[(&str, String)]) -> String {
    let body: Vec<String> = pairs.iter().map(|(k, v)| format!("{k}={v}")).collect();
    format!("\x1b]7501;{}\x1b\\", body.join(":"))
}

pub fn tmux_passthrough(seq: &str) -> String {
    format!("\x1bPtmux;{}\x1b\\", seq.replace('\x1b', "\x1b\x1b"))
}

fn b64_line(text: &str) -> Option<String> {
    let cleaned: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let line: String = cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(MSG_MAX_CHARS)
        .collect();
    (!line.is_empty()).then(|| base64::engine::general_purpose::STANDARD.encode(line))
}

struct Osc {
    tmux: bool,
}

impl Osc {
    fn from_env(env: &dyn Fn(&str) -> Option<String>, terminal: bool) -> Option<Self> {
        if !terminal {
            return None;
        }
        let off = ["0", "off", "false", "no"];
        if env("LIMBO_PROGRAM_STATUS")
            .is_some_and(|v| off.contains(&v.trim().to_lowercase().as_str()))
        {
            return None;
        }
        if env("TERM").as_deref() == Some("dumb") {
            return None;
        }
        Some(Self {
            tmux: env("TMUX").is_some_and(|t| !t.is_empty()),
        })
    }

    fn pairs(state: State, message: Option<&str>) -> Vec<(&'static str, String)> {
        let mut pairs = vec![
            ("state", state.as_str().to_string()),
            ("app", AGENT.to_string()),
        ];
        if state == State::Blocked {
            // Limbo's only blocked state is answering a question.
            pairs.push(("kind", "question".into()));
        }
        if let Some(m) = message.and_then(b64_line) {
            pairs.push(("msg", m));
        }
        pairs
    }

    fn send(&self, pairs: &[(&str, String)]) {
        let seq = encode_report(pairs);
        let out = if self.tmux {
            tmux_passthrough(&seq)
        } else {
            seq
        };
        let mut stdout = std::io::stdout().lock();
        let _ = stdout.write_all(out.as_bytes());
        let _ = stdout.flush();
    }
}

// ---------------------------------------------------------------------------

#[derive(Default)]
struct Inner {
    herdr: Option<Herdr>,
    osc: Option<Osc>,
    released: bool,
}

/// Fans lifecycle reports out to every active integration. Always safe to
/// call, even with none active.
#[derive(Default)]
pub struct Reporters {
    inner: Mutex<Inner>,
}

impl Reporters {
    /// Detect integrations. `terminal`: stdout is a real TTY the frontend
    /// lets us write escape sequences to (never in `--json` mode).
    pub fn from_env(terminal: bool) -> Self {
        let env = |k: &str| std::env::var(k).ok();
        Self {
            inner: Mutex::new(Inner {
                herdr: Herdr::from_env(&env),
                osc: Osc::from_env(&env, terminal),
                released: false,
            }),
        }
    }

    pub fn report(&self, state: State, message: Option<&str>) {
        let mut g = self.inner.lock().unwrap();
        if g.released {
            return;
        }
        if let Some(h) = &mut g.herdr {
            h.report(state, message);
        }
        if let Some(o) = &g.osc {
            o.send(&Osc::pairs(state, message));
        }
    }

    pub fn report_session(&self, id: &str, path: &Path) {
        let mut g = self.inner.lock().unwrap();
        if let Some(h) = &mut g.herdr {
            h.report_session(id, path);
        }
    }

    /// Clear our state on exit (idempotent).
    pub fn release(&self) {
        let mut g = self.inner.lock().unwrap();
        if std::mem::replace(&mut g.released, true) {
            return;
        }
        if let Some(h) = &mut g.herdr {
            h.release();
        }
        if let Some(o) = &g.osc {
            o.send(&[("state", "clear".into())]);
        }
    }

    pub fn active(&self) -> bool {
        let g = self.inner.lock().unwrap();
        g.herdr.is_some() || g.osc.is_some()
    }
}

/// Release on SIGHUP/SIGTERM too (a closed pane would otherwise keep
/// showing the agent), then die by the same signal.
pub fn install_exit_hooks(reporters: std::sync::Arc<Reporters>) {
    if !reporters.active() {
        return;
    }
    use tokio::signal::unix::{SignalKind, signal};
    for (kind, num) in [
        (SignalKind::hangup(), libc::SIGHUP),
        (SignalKind::terminate(), libc::SIGTERM),
    ] {
        let r = reporters.clone();
        let Ok(mut s) = signal(kind) else { continue };
        tokio::spawn(async move {
            if s.recv().await.is_some() {
                r.release();
                // SAFETY: restore the default action and re-raise.
                unsafe {
                    libc::signal(num, libc::SIG_DFL);
                    libc::raise(num);
                }
            }
        });
    }
}

/// Folds a run's events into the state reported when it ends: a clean
/// finish is `done`, an error `error`, a user interrupt plain `idle`.
#[derive(Debug, Clone)]
pub struct TurnOutcome {
    pub state: State,
    pub message: Option<String>,
}

impl Default for TurnOutcome {
    fn default() -> Self {
        Self {
            state: State::Done,
            message: None,
        }
    }
}

impl TurnOutcome {
    pub fn observe(&mut self, ev: &crate::agent::AgentEvent) {
        use crate::agent::AgentEvent;
        match ev {
            AgentEvent::Interrupted => {
                self.state = State::Idle;
                self.message = None;
            }
            AgentEvent::Error(m) if self.state != State::Idle => {
                self.state = State::Error;
                self.message = Some(m.clone());
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn osc_report_shape() {
        let pairs = Osc::pairs(State::Blocked, Some("等待\n确认"));
        let s = encode_report(&pairs);
        assert!(s.starts_with("\x1b]7501;state=blocked:app=limbo:kind=question:msg="));
        assert!(s.ends_with("\x1b\\"));
        let msg = pairs.iter().find(|(k, _)| *k == "msg").unwrap().1.clone();
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(msg)
            .unwrap();
        assert_eq!(String::from_utf8(decoded).unwrap(), "等待 确认");
        assert_eq!(tmux_passthrough("\x1bX"), "\x1bPtmux;\x1b\x1bX\x1b\\");
    }

    #[test]
    fn detection_needs_env() {
        let none = |_: &str| None;
        assert!(Herdr::from_env(&none).is_none());
        assert!(Osc::from_env(&none, false).is_none());
        assert!(Osc::from_env(&none, true).is_some());
        let off = |k: &str| (k == "LIMBO_PROGRAM_STATUS").then(|| "0".to_string());
        assert!(Osc::from_env(&off, true).is_none());
        let herdr = |k: &str| match k {
            "HERDR_ENV" => Some("1".into()),
            "HERDR_PANE_ID" => Some("p1".into()),
            "HERDR_BIN_PATH" => Some("/bin/true".into()),
            _ => None,
        };
        let mut h = Herdr::from_env(&herdr).unwrap();
        let a = h.next().parse::<u128>().unwrap();
        assert!(h.next().parse::<u128>().unwrap() > a);
    }
}
