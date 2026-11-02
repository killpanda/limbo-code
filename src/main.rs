//! Limbo — a terminal AI coding agent, rewritten in Rust.
//!
//! Module map (read them in this order to learn the codebase):
//! - `models`     — messages, tool calls, LLM events (the shared vocabulary)
//! - `config`     — `~/.limbo/config.toml`
//! - `llm`        — streaming clients for three wire dialects + the model catalog
//! - `tools`      — read / write / edit / bash / grep / find / ls
//! - `session`    — the JSONL journal (compatible with the Python version)
//! - `agent`      — the conversation loop that ties it all together
//! - `subagents`  — background subagents and the task board
//! - `tui`        — the ratatui + tachyonfx interface
//! - `headless`   — plain-terminal and JSON frontends

mod agent;
mod attachments;
mod auth;
mod background;
mod cli_auth;
mod commands;
mod compaction;
mod config;
mod goal;
mod headless;
mod history;
mod integrations;
mod learn;
mod llm;
mod memory;
mod models;
mod prompt;
mod pump;
mod recent_models;
mod session;
mod skills;
mod subagents;
mod tools;
mod trace;
mod trace_viewer;
mod tui;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::agent::Agent;
use crate::config::{Config, limbo_home};
use crate::llm::catalog;

#[derive(Parser, Debug)]
#[command(name = "limbo", version, about = "A terminal AI coding agent")]
struct Cli {
    /// Working directory (default: current directory)
    #[arg(long)]
    workdir: Option<PathBuf>,
    /// Directory for session JSONL files (default: ~/.limbo/sessions)
    #[arg(long)]
    session_dir: Option<PathBuf>,
    /// Resume the most recent session for the working directory
    #[arg(short = 'c', long = "continue", conflicts_with = "resume")]
    continue_session: bool,
    /// Resume a session by id (or unique id prefix)
    #[arg(short = 'r', long, value_name = "SESSION_ID")]
    resume: Option<String>,
    /// Override the model for this run (e.g. glm-4.7, kimi-k3:high)
    #[arg(short, long)]
    model: Option<String>,
    /// Plain-terminal REPL instead of the TUI
    #[arg(long)]
    headless: bool,
    /// With --headless (implied): print every event as one JSON line
    #[arg(long)]
    json: bool,
    /// Start in daily mode (non-coding assistant: web_fetch, memory) in the
    /// `[daily]` workdir; same as `limbo daily`
    #[arg(long)]
    daily: bool,
    /// Run one prompt non-interactively and exit
    #[arg(short, long, value_name = "PROMPT")]
    print: Option<String>,
    /// Config file (default: ~/.limbo/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List available models (optionally filtered)
    Models { query: Option<String> },
    /// Manage stored API keys (~/.limbo/auth.json)
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },
    /// List sessions for the working directory
    Sessions {
        /// Show sessions of every directory
        #[arg(long)]
        all: bool,
    },
}

#[derive(Subcommand, Debug)]
enum AuthAction {
    /// Store a provider's key (asks interactively when --key is missing)
    Login {
        provider: Option<String>,
        /// The key itself (default: prompt without echo)
        #[arg(long)]
        key: Option<String>,
        /// Custom endpoint (relay / private deployment)
        #[arg(long)]
        base_url: Option<String>,
        /// Skip the online check
        #[arg(long)]
        no_verify: bool,
    },
    /// Show where every provider's key comes from
    List,
    /// Delete a stored key
    Logout { provider: String },
}

fn main() {
    let code = match real_main() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e:#}");
            1
        }
    };
    std::process::exit(code);
}

fn real_main() -> anyhow::Result<i32> {
    // `limbo daily ...` is `limbo --daily ...`.
    let mut argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    if argv.get(1).is_some_and(|a| a == "daily") {
        argv[1] = "--daily".into();
    }
    let cli = Cli::parse_from(argv);
    let (mut config, warnings) = Config::load(cli.config.as_deref());
    for w in &warnings {
        eprintln!("warning: {w}");
    }
    if cli.daily && !config.daily.model.is_empty() && cli.model.is_none() {
        config.llm.model = config.daily.model.clone();
    }
    if let Some(m) = &cli.model {
        let (model, level) = catalog::split_model_arg(m);
        config.llm.model = model;
        if level.is_some() {
            config.llm.thinking_effort = level;
        }
    }
    let session_dir = cli
        .session_dir
        .clone()
        .unwrap_or_else(|| limbo_home().join("sessions"));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    match &cli.command {
        Some(Command::Models { query }) => {
            return Ok(cli_auth::list_models(&config, query.as_deref()));
        }
        Some(Command::Auth { action }) => {
            return Ok(match action {
                AuthAction::Login {
                    provider,
                    key,
                    base_url,
                    no_verify,
                } => rt.block_on(cli_auth::login(
                    provider.as_deref(),
                    key.clone(),
                    base_url.as_deref(),
                    !no_verify,
                )),
                AuthAction::List => cli_auth::list_logins(&config),
                AuthAction::Logout { provider } => cli_auth::logout(provider),
            });
        }
        _ => {}
    }
    let workdir = match &cli.workdir {
        Some(w) => std::fs::canonicalize(w)?,
        None if cli.daily => {
            let raw = &config.daily.workdir;
            let dir = match raw.strip_prefix("~/") {
                Some(rest) => dirs::home_dir().unwrap_or_default().join(rest),
                None if raw == "~" => dirs::home_dir().unwrap_or_default(),
                None => PathBuf::from(raw),
            };
            std::fs::create_dir_all(&dir).map_err(|e| {
                anyhow::anyhow!("cannot create daily workdir {}: {e}", dir.display())
            })?;
            std::fs::canonicalize(&dir)?
        }
        None => std::env::current_dir()?,
    };

    match cli.command {
        Some(Command::Models { .. }) | Some(Command::Auth { .. }) => unreachable!(),
        Some(Command::Sessions { all }) => {
            for m in session::list_sessions(&session_dir, (!all).then_some(workdir.as_path())) {
                println!(
                    "{}  {}  {}  {}",
                    m.id,
                    m.updated_at,
                    m.model,
                    if m.title.is_empty() {
                        "(untitled)"
                    } else {
                        &m.title
                    }
                );
            }
            return Ok(0);
        }
        None => {}
    }

    let resume = if cli.continue_session {
        Some(
            session::latest_session(&session_dir, Some(&workdir)).ok_or_else(|| {
                anyhow::anyhow!("No session to continue for {}", workdir.display())
            })?,
        )
    } else if let Some(id) = &cli.resume {
        Some(session::find_session(&session_dir, id)?)
    } else {
        None
    };

    // tokio's runtime is created by hand (instead of #[tokio::main]) so the
    // CLI parsing above stays plain synchronous code.
    rt.block_on(async move {
        let spec = catalog::resolve_model(&config.llm.model);
        if catalog::resolve_api_key(&spec, &config).is_none() {
            use std::io::IsTerminal;
            let tty = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
            if !(tty && cli.print.is_none() && cli_auth::first_run_login(&spec, &config).await) {
                eprintln!(
                    "Error: {}",
                    catalog::missing_api_key_message(&spec, &config)
                );
                return Ok(1);
            }
        }
        // `--daily` picks the posture of a *new* session only: a resumed
        // session keeps the mode it was saved with (/daily-mode switches it).
        let agent = match &resume {
            Some(p) => Agent::resume(config.clone(), p, &session_dir)?,
            None => Agent::create(config.clone(), &workdir, &session_dir, cli.daily)?,
        };
        if let Some(prompt) = cli.print {
            return Ok(headless::run_once(agent, prompt, cli.json).await);
        }
        if cli.headless || cli.json {
            return Ok(headless::run_repl(agent, cli.json).await);
        }
        tui::run(agent, config).await
    })
}
