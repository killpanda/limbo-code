//! The terminal UI: ratatui for widgets, tachyonfx for effects.
//!
//! The event loop multiplexes four sources with `tokio::select!`:
//! terminal input (keys, mouse, paste, resize), agent events, the running
//! turn handing the agent back, and an animation tick. It redraws only when
//! something changed or an animation is playing.

mod app;
mod chat;
mod fx;
mod input;
mod markdown;
mod text;
mod theme;
mod ui;

use std::io::stdout;
use std::time::{Duration, Instant};

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    supports_keyboard_enhancement,
};
use futures::StreamExt;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::agent::Agent;
use crate::config::Config;
use app::App;

fn restore_terminal(enhanced: bool) {
    let _ = disable_raw_mode();
    if enhanced {
        let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
    }
    let _ = execute!(
        stdout(),
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen,
        crossterm::cursor::Show
    );
}

pub async fn run(agent: Agent, config: Config) -> anyhow::Result<i32> {
    enable_raw_mode()?;
    let enhanced = supports_keyboard_enhancement().unwrap_or(false);
    execute!(
        stdout(),
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )?;
    if enhanced {
        // Lets us tell Shift+Enter from Enter on terminals that support the
        // kitty keyboard protocol.
        let _ = execute!(
            stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
    }
    // Restore the terminal even if we panic, or the user's shell is left in
    // raw mode.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal(enhanced);
        default_hook(info);
    }));

    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let result = event_loop(&mut terminal, App::new(agent, config)).await;
    restore_terminal(enhanced);
    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    mut app: App,
) -> anyhow::Result<i32> {
    let mut events = EventStream::new();
    let mut tick = tokio::time::interval(Duration::from_millis(33));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut last_frame = Instant::now();
    let mut last_anim = Instant::now();

    loop {
        if app.dirty || (app.animating() && last_anim.elapsed() >= Duration::from_millis(33)) {
            let elapsed = last_frame.elapsed();
            last_frame = Instant::now();
            last_anim = Instant::now();
            app.frame = app.frame.wrapping_add(1);
            terminal.draw(|f| {
                ui::draw(f, &mut app);
                let theme = app.theme;
                app.fx.process(elapsed, f.buffer_mut(), &theme);
            })?;
            app.dirty = false;
        }
        if app.quit {
            break;
        }
        if std::mem::take(&mut app.mouse_changed) {
            if app.mouse_capture {
                execute!(stdout(), EnableMouseCapture)?;
            } else {
                execute!(stdout(), DisableMouseCapture)?;
            }
        }

        tokio::select! {
            maybe = events.next() => match maybe {
                Some(Ok(Event::Key(k))) => app.on_key(k),
                Some(Ok(Event::Paste(p))) => app.on_paste(p),
                Some(Ok(Event::Mouse(m))) => app.on_mouse(m),
                Some(Ok(Event::Resize(_, _))) => app.dirty = true,
                Some(Ok(_)) => {}
                Some(Err(_)) | None => break,
            },
            Some(ev) = app.rx.recv() => {
                app.on_agent_event(ev);
                // Drain whatever else is queued so a fast stream renders in
                // batches instead of one frame per token.
                while let Ok(ev) = app.rx.try_recv() {
                    app.on_agent_event(ev);
                }
            }
            joined = async { app.job.as_mut().unwrap().await }, if app.job.is_some() => {
                match joined {
                    Ok(agent) => {
                        while let Ok(ev) = app.rx.try_recv() {
                            app.on_agent_event(ev);
                        }
                        app.job_done(agent);
                    }
                    Err(e) => anyhow::bail!("agent task failed: {e}"),
                }
            }
            _ = tick.tick() => {}
        }
    }

    // Let a running turn stop cleanly so its journal is consistent.
    app.interrupt();
    if let Some(job) = app.job.take()
        && let Ok(Ok(agent)) = tokio::time::timeout(Duration::from_secs(5), job).await
    {
        app.job_done(agent);
    }
    app.shutdown();
    Ok(0)
}
