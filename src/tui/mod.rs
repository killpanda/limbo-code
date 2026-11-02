//! The terminal UI: ratatui for widgets, tachyonfx for effects.
//!
//! The event loop multiplexes four sources with `tokio::select!`:
//! terminal input (keys, mouse, paste, resize), agent events, the running
//! turn handing the agent back, and an animation tick. It redraws only when
//! something changed or an animation is playing.

mod app;
mod chat;
mod clipboard;
mod fx;
mod game2048;
mod input;
mod markdown;
mod paste;
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
    BeginSynchronizedUpdate, EndSynchronizedUpdate, EnterAlternateScreen, LeaveAlternateScreen,
    disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement,
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

static ENHANCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn enhanced_flag() -> bool {
    ENHANCED.load(std::sync::atomic::Ordering::Relaxed)
}

fn enter_terminal(enhanced: bool) -> std::io::Result<()> {
    enable_raw_mode()?;
    execute!(
        stdout(),
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture,
        crossterm::cursor::Hide
    )?;
    if enhanced {
        // Lets us tell Shift+Enter from Enter on terminals that support the
        // kitty keyboard protocol.
        let _ = execute!(
            stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
    }
    Ok(())
}

pub async fn run(agent: Agent, config: Config) -> anyhow::Result<i32> {
    let enhanced = supports_keyboard_enhancement().unwrap_or(false);
    ENHANCED.store(enhanced, std::sync::atomic::Ordering::Relaxed);
    enter_terminal(enhanced)?;
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
            // Synchronized update: terminals that support it show the frame
            // all at once instead of cell by cell (others ignore it).
            execute!(stdout(), BeginSynchronizedUpdate)?;
            terminal.draw(|f| {
                ui::draw(f, &mut app);
                let theme = app.theme;
                app.fx.process(elapsed, f.buffer_mut(), &theme);
            })?;
            // The caret is drawn as a cell (ui::draw_input); the terminal
            // cursor stays hidden (ratatui hides it when the frame sets no
            // position) and only follows the caret so IMEs open there.
            if let Some(pos) = app.cursor {
                execute!(stdout(), crossterm::cursor::MoveTo(pos.x, pos.y))?;
            }
            execute!(stdout(), EndSynchronizedUpdate)?;
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
            Some(ev) = async { app.bg.as_mut().unwrap().recv().await }, if app.bg.is_some() => {
                app.on_background(ev);
            }
            item = async { app.btw_rx.as_mut().unwrap().recv().await }, if app.btw_rx.is_some() => {
                app.on_btw(item);
            }
            _ = tick.tick() => {}
        }
        if let Some(seq) = app.pending_osc52.take() {
            use std::io::Write;
            let _ = write!(stdout(), "{seq}");
            let _ = stdout().flush();
        }
        if let Some(path) = app.pending_editor.take() {
            // Hand the terminal to $EDITOR, then take it back.
            restore_terminal(enhanced_flag());
            let argv = crate::memory::editor_command(&path);
            let status = std::process::Command::new(&argv[0])
                .args(&argv[1..])
                .status();
            enter_terminal(enhanced_flag())?;
            terminal.clear()?;
            app.dirty = true;
            match status {
                Ok(s) if s.success() => app.toast("记忆已保存，从下一个会话起生效"),
                Ok(_) => app.toast("编辑器异常退出"),
                Err(e) => app.toast(format!("无法启动编辑器 {}：{e}", argv[0])),
            }
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
