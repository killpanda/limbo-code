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

/// Limbo Dawn: the warm sunrise palette daily mode switches to
/// (`[daily] look = "dawn"`): red → apricot → yellow.
pub const DAWN: Theme = Theme {
    name: "dawn",
    bg: rgb(21, 18, 26),
    panel: rgb(28, 24, 34),
    panel_hi: rgb(38, 32, 41),
    fg: rgb(238, 230, 225),
    dim: rgb(185, 173, 180),
    faint: rgb(142, 130, 144),
    accent: rgb(245, 160, 110),
    accent2: rgb(237, 88, 83),
    accent3: rgb(253, 231, 92),
    ok: rgb(150, 230, 140),
    warn: rgb(253, 200, 92),
    err: rgb(255, 99, 110),
    code: rgb(246, 201, 166),
    user: rgb(246, 201, 166),
    gradient: [rgb(237, 88, 83), rgb(245, 160, 110), rgb(253, 231, 92)],
};

pub const THEMES: [Theme; 3] = [NEON, AURORA, EMBER];

pub fn by_name(name: Option<&str>) -> Theme {
    if name == Some("omarchy")
        && let Some(t) = omarchy()
    {
        return t;
    }
    if name == Some("dawn") {
        return DAWN;
    }
    THEMES
        .iter()
        .copied()
        .find(|t| Some(t.name) == name)
        .unwrap_or(NEON)
}

// -- Omarchy ---------------------------------------------------------------------

fn hex(v: Option<&toml::Value>) -> Option<Color> {
    let s = v?.as_str()?.trim().trim_start_matches('#');
    let s = s.get(..6)?;
    let n = u32::from_str_radix(s, 16).ok()?;
    Some(rgb((n >> 16) as u8, (n >> 8) as u8, n as u8))
}

/// `rgba(26a269ee) rgba(2ec27eee) 45deg` → its colors.
fn parse_gradient(spec: Option<&toml::Value>) -> Vec<Color> {
    let Some(s) = spec.and_then(|v| v.as_str()) else {
        return vec![];
    };
    let re = regex::Regex::new(r"rgba?\(\s*([0-9A-Fa-f]{6})").unwrap();
    let mut out: Vec<Color> = re
        .captures_iter(s)
        .filter_map(|c| hex(Some(&toml::Value::String(c[1].to_string()))))
        .collect();
    if out.is_empty() {
        out = s
            .split_whitespace()
            .filter_map(|p| hex(Some(&toml::Value::String(p.into()))))
            .collect();
    }
    out
}

/// Follow the Omarchy system theme (`~/.local/state/omarchy/current/theme/
/// colors.toml`), with optional `limbo.toml` overrides next to it.
pub fn omarchy() -> Option<Theme> {
    let base = dirs::home_dir()?.join(".local/state/omarchy/current/theme");
    let colors: toml::Table = std::fs::read_to_string(base.join("colors.toml"))
        .ok()?
        .parse()
        .ok()?;
    let pick = |keys: &[&str]| keys.iter().find_map(|k| hex(colors.get(*k)));
    let bg = pick(&["background", "color0"])?;
    let fg = pick(&["foreground", "color7"])?;
    let accent = pick(&["accent", "blue", "color4"]).unwrap_or(fg);
    let yellow = pick(&["yellow", "color3"]).unwrap_or(accent);
    let red = pick(&["red", "color1"]).unwrap_or(accent);
    let green = pick(&["green", "color2"]).unwrap_or(accent);
    let magenta = pick(&["magenta", "color5"]).unwrap_or(accent);
    let orange = pick(&["orange", "yellow", "color3"]).unwrap_or(yellow);
    let mut grad = parse_gradient(colors.get("hyprland_active_border"));
    if grad.len() < 2 {
        grad = vec![accent, yellow];
    }
    let (from, to) = (grad[0], *grad.last().unwrap());
    let mut t = Theme {
        name: "omarchy",
        bg,
        panel: lerp(bg, fg, 0.05),
        panel_hi: lerp(bg, fg, 0.10),
        fg,
        dim: lerp(fg, bg, 0.34),
        faint: lerp(fg, bg, 0.52),
        accent,
        accent2: magenta,
        accent3: yellow,
        ok: green,
        warn: orange,
        err: red,
        code: orange,
        user: lerp(fg, accent, 0.45),
        gradient: [from, lerp(from, to, 0.5), to],
    };
    if let Ok(text) = std::fs::read_to_string(base.join("limbo.toml"))
        && let Ok(o) = text.parse::<toml::Table>()
    {
        let set = |slot: &mut Color, key: &str| {
            if let Some(c) = hex(o.get(key)) {
                *slot = c;
            }
        };
        set(&mut t.accent, "primary");
        set(&mut t.accent3, "accent");
        set(&mut t.ok, "success");
        set(&mut t.warn, "warning");
        set(&mut t.err, "error");
        set(&mut t.dim, "text-secondary");
        set(&mut t.faint, "text-tertiary");
        set(&mut t.user, "user-message-fg");
        set(&mut t.gradient[0], "grad-from");
        set(&mut t.gradient[1], "grad-mid");
        set(&mut t.gradient[2], "grad-to");
    }
    Some(t)
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
