//! Color themes and gradient helpers.
//!
//! Every color in the UI comes from a [`Theme`]; widgets never hard-code
//! RGB values. Three palettes ship: `neon` (cyan/magenta synthwave, the
//! default), `aurora` (teal/green northern lights) and `ember` (warm
//! orange/rose).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub name: &'static str,
    pub bg: Color,
    pub panel: Color,
    pub panel_hi: Color,
    pub fg: Color,
    pub dim: Color,
    pub faint: Color,
    pub accent: Color,
    pub accent2: Color,
    pub accent3: Color,
    pub ok: Color,
    pub warn: Color,
    pub err: Color,
    pub code: Color,
    pub user: Color,
    /// The wordmark/gauge gradient stops.
    pub gradient: [Color; 3],
}

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color::Rgb(r, g, b)
}

pub const NEON: Theme = Theme {
    name: "neon",
    bg: rgb(11, 12, 24),
    panel: rgb(20, 22, 40),
    panel_hi: rgb(32, 34, 60),
    fg: rgb(226, 230, 255),
    dim: rgb(140, 146, 190),
    faint: rgb(78, 82, 120),
    accent: rgb(94, 241, 255),
    accent2: rgb(255, 94, 219),
    accent3: rgb(155, 123, 255),
    ok: rgb(124, 255, 160),
    warn: rgb(255, 204, 102),
    err: rgb(255, 99, 132),
    code: rgb(255, 184, 108),
    user: rgb(255, 94, 219),
    gradient: [rgb(94, 241, 255), rgb(155, 123, 255), rgb(255, 94, 219)],
};

pub const AURORA: Theme = Theme {
    name: "aurora",
    bg: rgb(8, 16, 20),
    panel: rgb(14, 28, 34),
    panel_hi: rgb(22, 42, 50),
    fg: rgb(222, 246, 240),
    dim: rgb(130, 176, 170),
    faint: rgb(68, 100, 100),
    accent: rgb(80, 250, 180),
    accent2: rgb(120, 200, 255),
    accent3: rgb(190, 140, 255),
    ok: rgb(150, 255, 120),
    warn: rgb(255, 214, 120),
    err: rgb(255, 110, 120),
    code: rgb(255, 214, 120),
    user: rgb(120, 200, 255),
    gradient: [rgb(80, 250, 180), rgb(120, 200, 255), rgb(190, 140, 255)],
};

pub const EMBER: Theme = Theme {
    name: "ember",
    bg: rgb(22, 12, 12),
    panel: rgb(36, 20, 20),
    panel_hi: rgb(54, 30, 28),
    fg: rgb(255, 236, 226),
    dim: rgb(200, 160, 150),
    faint: rgb(110, 78, 72),
    accent: rgb(255, 160, 80),
    accent2: rgb(255, 90, 120),
    accent3: rgb(255, 210, 110),
    ok: rgb(170, 240, 120),
    warn: rgb(255, 210, 110),
    err: rgb(255, 90, 100),
    code: rgb(255, 210, 110),
    user: rgb(255, 120, 150),
    gradient: [rgb(255, 210, 110), rgb(255, 140, 80), rgb(255, 80, 120)],
};

pub const THEMES: [Theme; 3] = [NEON, AURORA, EMBER];

pub fn by_name(name: Option<&str>) -> Theme {
    THEMES
        .iter()
        .copied()
        .find(|t| Some(t.name) == name)
        .unwrap_or(NEON)
}

impl Theme {
    pub fn base(&self) -> Style {
        Style::default().fg(self.fg).bg(self.bg)
    }
}

fn channels(c: Color) -> (f32, f32, f32) {
    match c {
        Color::Rgb(r, g, b) => (r as f32, g as f32, b as f32),
        _ => (200.0, 200.0, 200.0),
    }
}

/// Linear interpolation between two RGB colors (`t` in 0..=1).
pub fn lerp(a: Color, b: Color, t: f32) -> Color {
    let (ar, ag, ab) = channels(a);
    let (br, bg, bb) = channels(b);
    let t = t.clamp(0.0, 1.0);
    rgb(
        (ar + (br - ar) * t) as u8,
        (ag + (bg - ag) * t) as u8,
        (ab + (bb - ab) * t) as u8,
    )
}

/// Sample a 3-stop gradient at `t` in 0..=1.
pub fn gradient_at(stops: &[Color; 3], t: f32) -> Color {
    if t < 0.5 {
        lerp(stops[0], stops[1], t * 2.0)
    } else {
        lerp(stops[1], stops[2], (t - 0.5) * 2.0)
    }
}

/// Color every character of `text` along the gradient, offset by `phase`
/// (animating `phase` makes the colors flow).
pub fn gradient_spans(
    text: &str,
    stops: &[Color; 3],
    phase: f32,
    modifier: Modifier,
) -> Vec<Span<'static>> {
    let n = text.chars().count().max(1) as f32;
    text.chars()
        .enumerate()
        .map(|(i, ch)| {
            let mut t = (i as f32 / n + phase).fract();
            // ping-pong so the wrap-around is seamless
            t = if t < 0.5 { t * 2.0 } else { 2.0 - t * 2.0 };
            Span::styled(
                ch.to_string(),
                Style::default()
                    .fg(gradient_at(stops, t))
                    .add_modifier(modifier),
            )
        })
        .collect()
}

/// The big landing wordmark.
pub const LOGO: [&str; 6] = [
    "██╗     ██╗███╗   ███╗██████╗  ██████╗ ",
    "██║     ██║████╗ ████║██╔══██╗██╔═══██╗",
    "██║     ██║██╔████╔██║██████╔╝██║   ██║",
    "██║     ██║██║╚██╔╝██║██╔══██╗██║   ██║",
    "███████╗██║██║ ╚═╝ ██║██████╔╝╚██████╔╝",
    "╚══════╝╚═╝╚═╝     ╚═╝╚═════╝  ╚═════╝ ",
];

pub fn logo_lines(theme: &Theme, phase: f32) -> Vec<Line<'static>> {
    LOGO.iter()
        .enumerate()
        .map(|(row, l)| {
            Line::from(gradient_spans(
                l,
                &theme.gradient,
                phase + row as f32 * 0.04,
                Modifier::BOLD,
            ))
        })
        .collect()
}

/// Braille spinner frames.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_endpoints() {
        assert_eq!(lerp(rgb(0, 0, 0), rgb(200, 100, 50), 0.0), rgb(0, 0, 0));
        assert_eq!(
            lerp(rgb(0, 0, 0), rgb(200, 100, 50), 1.0),
            rgb(200, 100, 50)
        );
    }
}
