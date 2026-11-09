//! Drawing: status bar, transcript, prompt box, popups.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Layout, Margin, Position, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};

use super::app::{Activity, App, COMMANDS, Popup, fmt_tokens};
use super::game2048;
use super::text::{ellipsize, str_width};
use super::theme::{SPINNER, Theme, gradient_at, gradient_spans, lerp, logo_lines};
use ratatui::widgets::Wrap;

pub fn draw(f: &mut Frame, app: &mut App) {
    let t = app.theme;
    let area = f.area();
    f.render_widget(Block::default().style(t.base()), area);

    let queued = app.queued();
    let input_width = area.width.saturating_sub(4) as usize;
    let (rows, _) = app.input.layout(input_width.max(1));
    let input_h = (rows.len().clamp(1, 8) + 2) as u16;
    let queue_h = queued.len().min(3) as u16;
    let [status, chat, queue, input, hint] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(queue_h),
        Constraint::Length(input_h),
        Constraint::Length(1),
    ])
    .areas(area);

    app.fx.areas.screen = area;
    app.fx.areas.status = status;
    app.fx.areas.chat = chat;
    app.fx.areas.input = input;

    draw_status(f, app, status);
    if app.entries.is_empty() {
        draw_landing(f, app, chat);
    } else {
        draw_chat(f, app, chat);
    }
    draw_queue(f, &t, &queued, queue);
    app.cursor = draw_input(f, app, input);
    draw_hint(f, app, hint);
    draw_commands(f, app, input);
    draw_toast(f, app, area);
    draw_popup(f, app, area);
}

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let t = &app.theme;
    let phase = if app.busy() {
        app.frame as f32 * 0.02
    } else {
        0.0
    };
    let mut left: Vec<Span> = vec![Span::raw(" ")];
    left.extend(gradient_spans(
        "◆ LIMBO",
        &t.gradient,
        phase,
        Modifier::BOLD,
    ));
    left.push(Span::styled("  ", Style::default()));
    let elapsed = app
        .busy_since
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let spin = SPINNER[app.frame % SPINNER.len()];
    let (chip, color) = match &app.activity {
        Activity::Idle => ("● ready".to_string(), t.ok),
        Activity::Thinking => (format!("{spin} thinking {elapsed:.0}s"), t.accent3),
        Activity::Writing => (format!("{spin} writing {elapsed:.0}s"), t.accent),
        Activity::Tool(name) => (format!("{spin} {name} {elapsed:.0}s"), t.accent2),
        Activity::Compacting => (format!("{spin} compacting {elapsed:.0}s"), t.warn),
    };
    left.push(Span::styled(
        chip,
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    let pill = |text: String, bg: ratatui::style::Color| {
        [
            Span::raw("  "),
            Span::styled(
                text,
                Style::default()
                    .fg(t.bg)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
            ),
        ]
    };
    if app.info.plan_mode {
        left.extend(pill(" PLAN ".into(), t.warn));
    }
    if app.info.daily_mode {
        left.extend(pill(" ☀ DAILY ".into(), t.accent3));
    }
    if app.info.code_mode {
        left.extend(pill(" ⌘ CODE ".into(), t.accent2));
    }
    if let Some(g) = &app.info.goal {
        left.extend(pill(format!(" {g} "), t.ok));
    }
    let tasks = app.tasks_label();
    if !tasks.is_empty() {
        left.extend(pill(format!(" ◆ {tasks} "), t.accent));
    }

    // Right side: model · context gauge · session tokens.
    let info = &app.info;
    let pct = if info.context_window > 0 {
        info.context_tokens as f64 / info.context_window as f64
    } else {
        0.0
    };
    let cells = 10usize;
    let filled = ((pct * cells as f64).round() as usize).min(cells);
    let mut right: Vec<Span> = Vec::new();
    let model = match &info.thinking {
        Some(l) => format!("{}:{l}", info.model),
        None => info.model.clone(),
    };
    right.push(Span::styled(model, Style::default().fg(t.dim)));
    right.push(Span::styled("  ", Style::default()));
    for i in 0..cells {
        let c = if i < filled {
            let gauge = if pct > 0.85 {
                [t.warn, t.err, t.err]
            } else {
                t.gradient
            };
            gradient_at(&gauge, i as f32 / cells as f32)
        } else {
            t.faint
        };
        right.push(Span::styled(
            if i < filled { "▰" } else { "▱" },
            Style::default().fg(c),
        ));
    }
    right.push(Span::styled(
        format!(
            " {:.0}% {}/{}",
            pct * 100.0,
            fmt_tokens(info.context_tokens),
            fmt_tokens(info.context_window)
        ),
        Style::default().fg(t.dim),
    ));
    right.push(Span::styled(
        format!("  Σ {} ", fmt_tokens(info.session_total)),
        Style::default().fg(t.faint),
    ));

    let lw: usize = left.iter().map(|s| str_width(&s.content)).sum();
    let rw: usize = right.iter().map(|s| str_width(&s.content)).sum();
    let mut spans = left;
    let total = area.width as usize;
    if lw + rw < total {
        let title = ellipsize(&info.title, total - lw - rw - 2);
        let pad = total - lw - rw - str_width(&title);
        let lp = pad / 2;
        spans.push(Span::raw(" ".repeat(lp)));
        spans.push(Span::styled(
            title,
            Style::default().fg(t.faint).add_modifier(Modifier::ITALIC),
        ));
        spans.push(Span::raw(" ".repeat(pad - lp)));
        spans.extend(right);
    }
    f.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(t.panel)),
        area,
    );
}

fn draw_landing(f: &mut Frame, app: &App, area: Rect) {
    let t = &app.theme;
    let phase = app.frame as f32 * 0.012;
    let mut lines: Vec<Line> = Vec::new();
    let logo = logo_lines(t, phase);
    let content_h = logo.len() + 12;
    let top = (area.height as usize).saturating_sub(content_h) / 2;
    lines.extend(std::iter::repeat_n(Line::default(), top));
    if area.width as usize >= 42 {
        lines.extend(logo);
    } else {
        lines.push(Line::from(gradient_spans(
            "◆ LIMBO",
            &t.gradient,
            phase,
            Modifier::BOLD,
        )));
    }
    lines.push(Line::default());
    let daily = app.info.daily_mode;
    lines.push(Line::from(Span::styled(
        if daily {
            "白天的事，也可以交给我。"
        } else {
            "terminal coding agent · rust edition"
        },
        Style::default().fg(t.dim).add_modifier(Modifier::ITALIC),
    )));
    lines.push(Line::default());
    // Pad every value to the same width so the centered block lines up.
    let rows = [
        ("model", app.info.model.clone()),
        (
            "workdir",
            ellipsize(&app.info.workdir, area.width.saturating_sub(14) as usize),
        ),
        ("session", app.info.session_id.clone()),
    ];
    let vw = rows.iter().map(|(_, v)| str_width(v)).max().unwrap_or(0);
    for (k, v) in rows {
        let pad = " ".repeat(vw - str_width(&v));
        lines.push(Line::from(vec![
            Span::styled(format!("{k:>8}  "), Style::default().fg(t.faint)),
            Span::styled(v, Style::default().fg(t.fg)),
            Span::raw(pad),
        ]));
    }
    lines.push(Line::default());
    let tips: [(&str, &str); 5] = if daily {
        [
            ("粘贴链接", "读网页"),
            ("/memory", "看看我记得什么"),
            ("/learn", "把经验沉淀成技能"),
            ("/", "命令"),
            ("ctrl+r", "历史会话"),
        ]
    } else {
        [
            ("/", "命令"),
            ("ctrl+r", "历史会话"),
            ("shift+tab", "思考强度"),
            ("ctrl+o", "展开工具输出"),
            ("esc", "打断"),
        ]
    };
    let mut tip_spans = Vec::new();
    for (i, (k, v)) in tips.iter().enumerate() {
        if i > 0 {
            tip_spans.push(Span::styled("  ·  ", Style::default().fg(t.faint)));
        }
        tip_spans.push(Span::styled(*k, Style::default().fg(t.accent)));
        tip_spans.push(Span::styled(format!(" {v}"), Style::default().fg(t.dim)));
    }
    lines.push(Line::from(tip_spans));
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), area);
}

fn draw_chat(f: &mut Frame, app: &mut App, area: Rect) {
    let t = app.theme;
    let inner = area.inner(Margin {
        horizontal: 2,
        vertical: 0,
    });
    let width = inner.width.saturating_sub(1) as usize;
    let (expand, thinking, frame) = (app.expand_tools, app.show_thinking, app.frame);
    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, e) in app.entries.iter_mut().enumerate() {
        if i > 0 {
            lines.push(Line::default());
        }
        lines.extend_from_slice(e.lines(&t, width, expand, thinking, frame));
    }
    lines.push(Line::default());
    let h = inner.height as usize;
    let total = lines.len();
    app.max_scroll = total.saturating_sub(h);
    app.scroll = app.scroll.min(app.max_scroll);
    let start = total.saturating_sub(h + app.scroll);
    let visible: Vec<Line> = lines.into_iter().skip(start).take(h).collect();
    f.render_widget(Paragraph::new(visible), inner);

    if app.max_scroll > 0 {
        let mut state = ScrollbarState::new(app.max_scroll).position(app.max_scroll - app.scroll);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("┃")
                .track_style(Style::default().fg(t.panel_hi))
                .thumb_style(Style::default().fg(t.accent3)),
            area,
            &mut state,
        );
    }
    if app.scroll > 0 {
        let label = format!(" ↓ {} 行 · ctrl+end 回到底部 ", app.scroll);
        let w = str_width(&label) as u16;
        let r = Rect {
            x: area.x + area.width.saturating_sub(w + 2),
            y: area.y + area.height.saturating_sub(1),
            width: w.min(area.width),
            height: 1,
        };
        f.render_widget(
            Paragraph::new(Span::styled(label, Style::default().fg(t.bg).bg(t.accent3))),
            r,
        );
    }
}

fn draw_queue(f: &mut Frame, t: &Theme, queued: &[String], area: Rect) {
    if area.height == 0 {
        return;
    }
    let lines: Vec<Line> = queued
        .iter()
        .take(area.height as usize)
        .map(|q| {
            Line::from(vec![
                Span::styled("  ⏳ ", Style::default().fg(t.warn)),
                Span::styled(
                    ellipsize(&q.replace('\n', " "), area.width.saturating_sub(8) as usize),
                    Style::default().fg(t.dim).add_modifier(Modifier::ITALIC),
                ),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(lines), area);
}

/// Draws the prompt box and returns where the caret is (None while a popup
/// has the focus).
///
/// The caret is painted into the buffer as a reversed cell instead of using
/// the terminal's own blinking cursor. The screen redraws up to 30 times a
/// second while something animates (the landing logo, the spinner), and
/// every frame would re-show and move the terminal cursor: many terminals
/// restart the blink on each move, so it flickered fast, and it was briefly
/// visible wherever ratatui was writing cells. The terminal cursor stays
/// hidden; the event loop still moves it to the returned position so IME
/// candidate windows open at the caret.
fn draw_input(f: &mut Frame, app: &App, area: Rect) -> Option<Position> {
    let t = &app.theme;
    let border = if app.busy() {
        let w = ((app.frame as f32 * 0.12).sin() + 1.0) / 2.0;
        lerp(t.accent, t.accent2, w)
    } else {
        t.accent3
    };
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .title(Line::from(vec![Span::styled(
            " ❯ ",
            Style::default().fg(border).add_modifier(Modifier::BOLD),
        )]));
    if app.info.plan_mode {
        block = block.title_top(
            Line::from(Span::styled(
                " plan · read-only ",
                Style::default().fg(t.warn),
            ))
            .right_aligned(),
        );
    }
    if !app.queued().is_empty() {
        block = block.title_bottom(
            Line::from(Span::styled(
                format!(" {} 条排队 · ↑ 撤回 ", app.queued().len()),
                Style::default().fg(t.warn),
            ))
            .right_aligned(),
        );
    }
    let inner = block.inner(area).inner(Margin {
        horizontal: 1,
        vertical: 0,
    });
    f.render_widget(block, area);
    if app.input.is_empty() {
        let ph = if app.busy() {
            "输入的消息会排队到下一步 · esc 打断"
        } else {
            "问点什么，或输入 / 查看命令…"
        };
        f.render_widget(
            Paragraph::new(Span::styled(
                ph,
                Style::default().fg(t.faint).add_modifier(Modifier::ITALIC),
            )),
            inner,
        );
        return caret(f, app, Position::new(inner.x, inner.y));
    }
    let (rows, (cr, cc)) = app.input.layout(inner.width as usize);
    let h = inner.height as usize;
    let first = cr.saturating_sub(h.saturating_sub(1));
    let lines: Vec<Line> = rows
        .iter()
        .skip(first)
        .take(h)
        .map(|r| Line::from(Span::styled(r.clone(), Style::default().fg(t.fg))))
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
    caret(
        f,
        app,
        Position::new(inner.x + cc as u16, inner.y + (cr - first) as u16),
    )
}

/// Paints the caret cell (see [`draw_input`]) and returns its position.
fn caret(f: &mut Frame, app: &App, pos: Position) -> Option<Position> {
    if app.popup.is_some() || !f.area().contains(pos) {
        return None;
    }
    if let Some(cell) = f.buffer_mut().cell_mut(pos) {
        cell.set_style(Style::default().add_modifier(Modifier::REVERSED));
    }
    Some(pos)
}

fn draw_hint(f: &mut Frame, app: &App, area: Rect) {
    let t = &app.theme;
    let keys: &[(&str, &str)] = if app.busy() {
        &[
            ("esc", "打断"),
            ("enter", "插话"),
            ("alt+enter", "任务后再发"),
            ("ctrl+o", "工具"),
            ("ctrl+t", "思考"),
            ("pgup", "滚动"),
        ]
    } else {
        &[
            ("enter", "发送"),
            ("shift+enter", "换行"),
            ("/", "命令"),
            ("ctrl+v", "贴图"),
            ("ctrl+r", "会话"),
            ("ctrl+o", "工具"),
            ("ctrl+t", "思考"),
            ("ctrl+c", "退出"),
        ]
    };
    let mut spans = vec![Span::raw(" ")];
    for (i, (k, v)) in keys.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", Style::default().fg(t.faint)));
        }
        spans.push(Span::styled(*k, Style::default().fg(t.dim)));
        spans.push(Span::styled(format!(" {v}"), Style::default().fg(t.faint)));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_commands(f: &mut Frame, app: &mut App, input: Rect) {
    let matches = app.command_matches();
    if matches.is_empty() || app.popup.is_some() {
        return;
    }
    let t = app.theme;
    let n = matches.len().min(10);
    app.cmd_selected = app.cmd_selected.min(matches.len() - 1);
    let h = n as u16 + 2;
    let w = input.width.min(76);
    let area = Rect {
        x: input.x,
        y: input.y.saturating_sub(h),
        width: w,
        height: h,
    };
    f.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.faint))
        .style(Style::default().bg(t.panel));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let skip = app.cmd_selected.saturating_sub(n - 1);
    let lines: Vec<Line> = matches
        .iter()
        .enumerate()
        .skip(skip)
        .take(n)
        .map(|(i, (name, args, help))| {
            let sel = i == app.cmd_selected;
            let bg = if sel { t.panel_hi } else { t.panel };
            let name_s = format!(" /{name:<10}");
            let args_s = format!("{args:<20}");
            let help_s = ellipsize(
                help,
                (inner.width as usize).saturating_sub(str_width(&name_s) + 23),
            );
            Line::from(vec![
                Span::styled(
                    if sel { "▸" } else { " " },
                    Style::default().fg(t.accent).bg(bg),
                ),
                Span::styled(
                    name_s,
                    Style::default()
                        .fg(if sel { t.accent } else { t.fg })
                        .bg(bg)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(args_s, Style::default().fg(t.faint).bg(bg)),
                Span::styled(format!(" {help_s}"), Style::default().fg(t.dim).bg(bg)),
            ])
        })
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_toast(f: &mut Frame, app: &mut App, area: Rect) {
    let Some((text, at)) = &app.toast else { return };
    let age = at.elapsed().as_secs_f32();
    if age > 2.5 {
        app.toast = None;
        return;
    }
    let t = &app.theme;
    let fade = if age > 2.0 { (age - 2.0) * 2.0 } else { 0.0 };
    let w = (str_width(text) + 4) as u16;
    let r = Rect {
        x: area.x + area.width.saturating_sub(w + 1),
        y: area.y + 1,
        width: w.min(area.width),
        height: 1,
    };
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ✦ ", Style::default().fg(lerp(t.accent2, t.panel, fade))),
            Span::styled(
                format!("{text} "),
                Style::default().fg(lerp(t.fg, t.panel, fade)),
            ),
        ]))
        .style(Style::default().bg(t.panel_hi)),
        r,
    );
}

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width.saturating_sub(2));
    let h = h.min(area.height.saturating_sub(2));
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn popup_block<'a>(t: &Theme, title: &str, footer: &str) -> Block<'a> {
    let mut title_spans = vec![Span::raw(" ")];
    title_spans.extend(gradient_spans(title, &t.gradient, 0.0, Modifier::BOLD));
    title_spans.push(Span::raw(" "));
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(t.accent3))
        .title(Line::from(title_spans))
        .title_bottom(
            Line::from(Span::styled(
                format!(" {footer} "),
                Style::default().fg(t.faint),
            ))
            .right_aligned(),
        )
        .style(Style::default().bg(t.panel).fg(t.fg))
}

fn draw_popup(f: &mut Frame, app: &mut App, area: Rect) {
    let t = app.theme;
    let Some(popup) = &app.popup else { return };
    let rect = match popup {
        Popup::Help => {
            let r = centered(area, 72, (COMMANDS.len() + 14) as u16);
            f.render_widget(Clear, r);
            let block = popup_block(&t, "Limbo · 帮助", "任意键关闭");
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            let mut lines = vec![Line::from(Span::styled(
                "命令",
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            ))];
            for c in COMMANDS {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("  /{:<10}", c.name),
                        Style::default().fg(t.fg).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!("{:<20}", c.args), Style::default().fg(t.faint)),
                    Span::styled(c.help, Style::default().fg(t.dim)),
                ]));
            }
            lines.push(Line::default());
            lines.push(Line::from(Span::styled(
                "按键",
                Style::default().fg(t.accent).add_modifier(Modifier::BOLD),
            )));
            for (k, v) in [
                ("enter / shift+enter", "发送 / 换行（忙时 enter 排队）"),
                ("esc · ctrl+c", "打断当前任务 · 连按两次退出"),
                ("↑ ↓", "历史输入；空输入时 ↑ 撤回排队消息"),
                ("pgup pgdn · 滚轮", "滚动对话"),
                ("ctrl+o · ctrl+t", "展开工具输出 · 显示思考过程"),
                ("shift+tab", "轮换思考强度"),
                ("ctrl+r · ctrl+l", "历史会话 · 清屏"),
            ] {
                lines.push(Line::from(vec![
                    Span::styled(format!("  {k:<22}"), Style::default().fg(t.fg)),
                    Span::styled(v, Style::default().fg(t.dim)),
                ]));
            }
            f.render_widget(Paragraph::new(lines), inner);
            r
        }
        Popup::Sessions {
            items,
            selected,
            all,
        } => {
            let r = centered(area, 96, 22);
            f.render_widget(Clear, r);
            let title = if *all {
                "历史会话 · 全部目录"
            } else {
                "历史会话 · 当前目录"
            };
            let block = popup_block(&t, title, "↑↓ 选择 · enter 打开 · tab 切换范围 · esc 关闭");
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            let h = inner.height as usize;
            let skip = selected.saturating_sub(h.saturating_sub(1));
            let lines: Vec<Line> = if items.is_empty() {
                vec![Line::from(Span::styled(
                    "（还没有会话）",
                    Style::default().fg(t.faint),
                ))]
            } else {
                items
                    .iter()
                    .enumerate()
                    .skip(skip)
                    .take(h)
                    .map(|(i, m)| {
                        let sel = i == *selected;
                        let bg = if sel { t.panel_hi } else { t.panel };
                        let when = m
                            .updated_at
                            .get(..16)
                            .unwrap_or(&m.updated_at)
                            .replace('T', " ");
                        let title = if m.title.is_empty() {
                            "(untitled)".to_string()
                        } else {
                            m.title.clone()
                        };
                        let w = inner.width as usize;
                        Line::from(vec![
                            Span::styled(
                                if sel { "▸ " } else { "  " },
                                Style::default().fg(t.accent).bg(bg),
                            ),
                            Span::styled(format!("{when}  "), Style::default().fg(t.faint).bg(bg)),
                            Span::styled(
                                ellipsize(&title, w.saturating_sub(42)),
                                Style::default()
                                    .fg(if sel { t.accent } else { t.fg })
                                    .bg(bg),
                            ),
                            Span::styled(
                                format!("  {}", ellipsize(&m.model, 18)),
                                Style::default().fg(t.dim).bg(bg),
                            ),
                        ])
                    })
                    .collect()
            };
            f.render_widget(Paragraph::new(lines), inner);
            r
        }
        Popup::Models {
            items,
            selected,
            filter,
        } => {
            let r = centered(area, 70, 24);
            f.render_widget(Clear, r);
            let block = popup_block(
                &t,
                "切换模型",
                "输入过滤 · enter 选择（可直接输入任意模型 id）· esc",
            );
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            let [search, list] =
                Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).areas(inner);
            f.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled("⌕ ", Style::default().fg(t.accent)),
                        Span::styled(filter.clone(), Style::default().fg(t.fg)),
                        Span::styled("▏", Style::default().fg(t.accent)),
                    ]),
                    Line::from(Span::styled(
                        "─".repeat(inner.width as usize),
                        Style::default().fg(t.faint),
                    )),
                ]),
                search,
            );
            let visible = crate::tui::app::visible_models(items, filter);
            let sel = (*selected).min(visible.len().saturating_sub(1));
            let h = list.height as usize;
            let skip = sel.saturating_sub(h.saturating_sub(1));
            let lines: Vec<Line> = visible
                .iter()
                .enumerate()
                .skip(skip)
                .take(h)
                .map(|(i, (id, provider))| {
                    let s = i == sel;
                    let bg = if s { t.panel_hi } else { t.panel };
                    let current = *id == app.info.model;
                    Line::from(vec![
                        Span::styled(
                            if s { "▸ " } else { "  " },
                            Style::default().fg(t.accent).bg(bg),
                        ),
                        Span::styled(
                            format!("{id:<34}"),
                            Style::default()
                                .fg(if s { t.accent } else { t.fg })
                                .bg(bg)
                                .add_modifier(if current {
                                    Modifier::BOLD
                                } else {
                                    Modifier::empty()
                                }),
                        ),
                        Span::styled(provider.clone(), Style::default().fg(t.dim).bg(bg)),
                        Span::styled(
                            if current { "  ← 当前" } else { "" },
                            Style::default().fg(t.ok).bg(bg),
                        ),
                    ])
                })
                .collect();
            f.render_widget(Paragraph::new(lines), list);
            r
        }
        Popup::VerifyPicker { items, selected } => {
            let r = centered(area, 80, (items.len() + 6) as u16);
            f.render_widget(Clear, r);
            let block = popup_block(&t, "🎯 确认验收方式", "↑↓ 选择 · enter 确认 · esc 跳过");
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            let mut lines = vec![
                Line::from(Span::styled(
                    "模型提议用下面的命令判断目标是否达成（退出码 0 = 通过）：",
                    Style::default().fg(t.dim),
                )),
                Line::default(),
            ];
            for (i, item) in items.iter().enumerate() {
                let sel = i == *selected;
                let bg = if sel { t.panel_hi } else { t.panel };
                let is_cmd = i + 2 < items.len();
                lines.push(Line::from(vec![
                    Span::styled(
                        if sel { "▸ " } else { "  " },
                        Style::default().fg(t.accent).bg(bg),
                    ),
                    Span::styled(
                        ellipsize(item, inner.width.saturating_sub(4) as usize),
                        Style::default()
                            .fg(if sel {
                                t.accent
                            } else if is_cmd {
                                t.code
                            } else {
                                t.dim
                            })
                            .bg(bg)
                            .add_modifier(if is_cmd {
                                Modifier::BOLD
                            } else {
                                Modifier::empty()
                            }),
                    ),
                ]));
            }
            f.render_widget(Paragraph::new(lines), inner);
            r
        }
        Popup::Auth {
            provider,
            model,
            input,
            status,
            checking,
        } => {
            let r = centered(area, 64, 9);
            f.render_widget(Clear, r);
            let name = crate::llm::catalog::provider_display_name(provider);
            let footer = if *checking {
                "验证中… · esc 取消"
            } else {
                "enter 验证并保存 · esc 取消"
            };
            let block = popup_block(&t, &format!("🔑 登录 {name}"), footer);
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            // Never draw the key itself: a masked preview is enough.
            let masked = if input.is_empty() {
                String::new()
            } else {
                crate::auth::mask(input)
            };
            let lines = vec![
                Line::from(Span::styled(
                    format!("模型 {model} 需要 {name} 的 API key，保存到 ~/.limbo/auth.json"),
                    Style::default().fg(t.dim),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled("key ❯ ", Style::default().fg(t.accent)),
                    Span::styled(masked, Style::default().fg(t.fg)),
                    Span::styled("▍", Style::default().fg(t.accent)),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    status.clone().unwrap_or_default(),
                    Style::default().fg(if status.as_deref().is_some_and(|s| s.starts_with('✗')) {
                        t.err
                    } else {
                        t.dim
                    }),
                )),
            ];
            f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
            r
        }
        Popup::Btw {
            question,
            answer,
            done,
            scroll,
        } => {
            let r = centered(area, 90, area.height.saturating_sub(6).max(10));
            f.render_widget(Clear, r);
            let footer = if *done {
                "任意键关闭（答案不会进入会话）"
            } else {
                "回答中… · 任意键关闭（答案不会进入会话）"
            };
            let block = popup_block(&t, &format!("💬 btw · {}", ellipsize(question, 60)), footer);
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            let body = if answer.is_empty() && !*done {
                format!(
                    "{} 思考中…",
                    super::theme::SPINNER[app.frame % super::theme::SPINNER.len()]
                )
            } else {
                answer.clone()
            };
            let lines =
                super::markdown::render(&body, &t, inner.width as usize, Style::default().fg(t.fg));
            f.render_widget(
                Paragraph::new(lines)
                    .wrap(Wrap { trim: false })
                    .scroll((*scroll, 0)),
                inner,
            );
            r
        }
        Popup::Text {
            title,
            body,
            scroll,
        } => {
            let h = (body.lines().count() as u16 + 4).clamp(6, area.height.saturating_sub(4));
            let r = centered(area, 100, h);
            f.render_widget(Clear, r);
            let block = popup_block(&t, title, "↑↓ 滚动 · 任意键关闭");
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            f.render_widget(
                Paragraph::new(body.clone())
                    .style(Style::default().fg(t.fg))
                    .wrap(Wrap { trim: false })
                    .scroll((*scroll, 0)),
                inner,
            );
            r
        }
        Popup::Game(game) => {
            let r = centered(area, 4 * 9 + 6, 4 * 4 + 6);
            f.render_widget(Clear, r);
            let title = format!("2048 · 分数 {}", game.score);
            let footer = if game.over() {
                "游戏结束 · R 重开 · Esc 关闭"
            } else if game.won {
                "🎉 2048！继续玩 · R 重开 · Esc"
            } else {
                "方向键/WASD · R 重开 · Esc 关闭"
            };
            let block = popup_block(&t, &title, footer);
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 1,
            });
            f.render_widget(block, r);
            let mut lines = vec![];
            for row in game.grid.iter() {
                for sub in 0..4 {
                    let mut spans = vec![];
                    for v in row.iter() {
                        let level = if *v == 0 {
                            0.0
                        } else {
                            (v.trailing_zeros() as f32 / 11.0).min(1.0)
                        };
                        let bg = if *v == 0 {
                            t.panel_hi
                        } else {
                            gradient_at(&t.gradient, level)
                        };
                        let text = if sub == 1 && *v != 0 {
                            format!("{v:^8}")
                        } else {
                            " ".repeat(8)
                        };
                        let text = if sub == 3 { " ".repeat(8) } else { text };
                        let style = if sub == 3 {
                            Style::default().bg(t.panel)
                        } else {
                            Style::default()
                                .fg(t.bg)
                                .bg(bg)
                                .add_modifier(Modifier::BOLD)
                        };
                        spans.push(Span::styled(text, style));
                        spans.push(Span::styled(" ", Style::default().bg(t.panel)));
                    }
                    lines.push(Line::from(spans));
                }
            }
            let _ = game2048::SIZE;
            f.render_widget(Paragraph::new(lines), inner);
            r
        }
        Popup::Agents { selected, scroll } => {
            let r = centered(
                area,
                area.width.saturating_sub(6),
                area.height.saturating_sub(4),
            );
            f.render_widget(Clear, r);
            let block = popup_block(
                &t,
                "◆ 子代理",
                "↑↓ 选择 · a 批准 · n 拒绝 · p 暂停/恢复 · x 取消 · r 重试 · @ 发消息 · 其他键关闭",
            );
            let inner = block.inner(r).inner(Margin {
                horizontal: 1,
                vertical: 0,
            });
            f.render_widget(block, r);
            let Some(sup) = app.sup.clone() else { return };
            let tasks = sup.tasks();
            let [list, detail] =
                Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
                    .areas(inner);
            let sel = (*selected).min(tasks.len().saturating_sub(1));
            let mut lines: Vec<Line> = vec![];
            if tasks.is_empty() {
                lines.push(Line::from(Span::styled(
                    "任务板是空的。让主 agent 用 plan_tasks 派活，或 /agents scan 看项目画像。",
                    Style::default().fg(t.faint),
                )));
            }
            for (i, task) in tasks.iter().enumerate() {
                let s_ = i == sel;
                let bg = if s_ { t.panel_hi } else { t.panel };
                let color = match task.status.as_str() {
                    "running" => t.accent,
                    "done" => t.ok,
                    "failed" => t.err,
                    "waiting" | "proposed" | "paused" => t.warn,
                    _ => t.dim,
                };
                let icon = match task.status.as_str() {
                    "running" => super::theme::SPINNER[app.frame % super::theme::SPINNER.len()],
                    "done" => "✓",
                    "failed" => "✗",
                    "cancelled" => "⊘",
                    "waiting" => "?",
                    "paused" => "⏸",
                    "proposed" => "◇",
                    _ => "○",
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("{icon} "), Style::default().fg(color).bg(bg)),
                    Span::styled(
                        format!("{:<4}", task.id),
                        Style::default().fg(t.faint).bg(bg),
                    ),
                    Span::styled(
                        ellipsize(
                            &format!("{} · {}", task.profile, task.title),
                            list.width.saturating_sub(16) as usize,
                        ),
                        Style::default().fg(if s_ { t.accent } else { t.fg }).bg(bg),
                    ),
                    Span::styled(
                        format!(
                            " {}",
                            crate::subagents::supervisor::status_label(&task.status)
                        ),
                        Style::default().fg(color).bg(bg),
                    ),
                ]));
                let act = sup.activity(&task.id);
                if task.status == "running" && !act.is_empty() {
                    lines.push(Line::from(Span::styled(
                        format!("     {act}"),
                        Style::default().fg(t.faint).bg(bg),
                    )));
                }
            }
            f.render_widget(Paragraph::new(lines), list);
            if let Some(task) = tasks.get(sel) {
                let mut body = sup.status_text(Some(&task.id));
                let transcript = sup.transcript(&task.id, 30);
                if !transcript.is_empty() {
                    body.push_str("\n\n── 记录 ──");
                    for (role, text) in transcript {
                        let first: String = text
                            .lines()
                            .next()
                            .unwrap_or("")
                            .chars()
                            .take(200)
                            .collect();
                        body.push_str(&format!("\n[{role}] {first}"));
                    }
                }
                f.render_widget(
                    Paragraph::new(body)
                        .style(Style::default().fg(t.dim))
                        .wrap(Wrap { trim: false })
                        .scroll((*scroll, 0))
                        .block(
                            Block::default()
                                .borders(Borders::LEFT)
                                .border_style(Style::default().fg(t.faint)),
                        ),
                    detail,
                );
            }
            r
        }
    };
    app.fx.popup_area(rect, &t);
}
