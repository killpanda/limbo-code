//! Limbo — a terminal AI coding agent, rewritten in Rust.
//!
//! Module map (read them in this order to learn the codebase):
//! - `models`     — messages, tool calls, LLM events (the shared vocabulary)
//! - `config`     — `~/.limbo/config.toml`
//! - `llm`        — streaming clients for three wire dialects + the model catalog
//! - `tools`      — read / write / edit / bash / grep / find / ls
//! - `session`    — the JSONL journal (compatible with the Python version)
//! - `agent`      — the conversation loop that ties it all together
//! - `subagent`   — the `task` tool (child agents)
//! - `tui`        — the ratatui + tachyonfx interface
//! - `headless`   — plain-terminal and JSON frontends

mod agent;
mod auth;
mod compaction;
mod config;
mod headless;
mod history;
mod llm;
mod models;
mod prompt;
mod session;
mod subagent;
mod tools;
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
    /// List the built-in model catalog
    Models,
    /// List sessions for the working directory
    Sessions {
        /// Show sessions of every directory
        #[arg(long)]
        all: bool,
    },
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
    let cli = Cli::parse();
    let (mut config, warnings) = Config::load(cli.config.as_deref());
    for w in &warnings {
        eprintln!("warning: {w}");
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
    let workdir = match &cli.workdir {
        Some(w) => std::fs::canonicalize(w)?,
        None => std::env::current_dir()?,
    };

    match cli.command {
        Some(Command::Models) => {
            print_models(&config);
            return Ok(0);
        }
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
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let agent = match &resume {
            Some(p) => Agent::resume(config.clone(), p, &session_dir)?,
            None => Agent::new(config.clone(), &workdir, &session_dir)?,
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

fn print_models(config: &Config) {
    let current = &config.llm.model;
    let mut last_provider = "";
    let mut specs: Vec<_> = catalog::catalog().values().collect();
    specs.sort_by_key(|s| (s.provider.id, s.id.clone()));
    for spec in specs {
        if spec.provider.id != last_provider {
            last_provider = spec.provider.id;
            let has_key = catalog::resolve_api_key(spec, config).is_some();
            println!(
                "\n{} {}",
                catalog::provider_display_name(spec.provider.id),
                if has_key { "✓" } else { "（无 key）" }
            );
        }
        let mark = if &spec.id == current { "●" } else { " " };
        println!(
            "  {mark} {:<32} ctx {:>9}  {}",
            spec.id,
            spec.context_window,
            if spec.vision { "vision" } else { "" }
        );
    }
}
