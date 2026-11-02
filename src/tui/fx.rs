//! Visual effects (tachyonfx): the screen materializing on startup, popups
//! sweeping in, a glow on the prompt when a turn ends, a red flash on
//! errors, a hue ripple when context is compacted.
//!
//! tachyonfx effects are post-processing passes: after widgets render into
//! the frame buffer, each running effect rewrites the cells in its area
//! (colors, characters) according to its timer. Effects need screen areas,
//! so the drawing code reports the latest layout via [`Fx::areas`].

use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use tachyonfx::{EffectManager, Interpolation, Motion, fx};

use super::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionLevel {
    Full,
    Reduced,
    Off,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Areas {
    pub screen: Rect,
    pub status: Rect,
    pub chat: Rect,
    pub input: Rect,
}

pub struct Fx {
    manager: EffectManager<&'static str>,
    pub level: MotionLevel,
    pub areas: Areas,
    /// Set when a popup opened; the draw code adds the effect once it knows
    /// the popup's rectangle.
    pub popup_pending: bool,
    startup_pending: bool,
}

impl Fx {
    pub fn new(motion: Option<&str>) -> Self {
        let level = match motion.map(str::to_lowercase).as_deref() {
            Some("off" | "none" | "0") => MotionLevel::Off,
            Some("reduced") => MotionLevel::Reduced,
            Some(_) => MotionLevel::Full,
            None if std::env::var("SSH_CONNECTION").is_ok() => MotionLevel::Reduced,
            None => MotionLevel::Full,
        };
        Self {
            manager: EffectManager::default(),
            level,
            areas: Areas::default(),
            popup_pending: false,
            startup_pending: true,
        }
    }

    pub fn running(&self) -> bool {
        self.level != MotionLevel::Off
            && (self.manager.is_running() || self.popup_pending || self.startup_pending)
    }

    fn on(&self) -> bool {
        self.level != MotionLevel::Off
    }

    pub fn startup(&mut self, _theme: &Theme) {
        self.startup_pending = true;
    }

    fn add_startup(&mut self, theme: &Theme) {
        if self.level == MotionLevel::Full {
            let e = fx::parallel(&[
                fx::coalesce((900, Interpolation::QuadOut)),
                fx::sweep_in(
                    Motion::LeftToRight,
                    40,
                    10,
                    theme.bg,
                    (700, Interpolation::QuadOut),
                ),
            ]);
            self.manager
                .add_unique_effect("startup", e.with_area(self.areas.screen));
        } else if self.level == MotionLevel::Reduced {
            self.manager.add_unique_effect(
                "startup",
                fx::fade_from_fg(theme.bg, (300, Interpolation::QuadOut))
                    .with_area(self.areas.screen),
            );
        }
    }

    pub fn popup_opened(&mut self) {
        self.popup_pending = self.on();
    }

    pub fn popup_area(&mut self, area: Rect, theme: &Theme) {
        if !std::mem::take(&mut self.popup_pending) {
            return;
        }
        let e = if self.level == MotionLevel::Full {
            fx::parallel(&[
                fx::coalesce((280, Interpolation::QuadOut)),
                fx::sweep_in(
                    Motion::UpToDown,
                    6,
                    0,
                    theme.bg,
                    (260, Interpolation::QuadOut),
                ),
            ])
        } else {
            fx::fade_from_fg(theme.bg, (150, Interpolation::Linear))
        };
        self.manager.add_unique_effect("popup", e.with_area(area));
    }

    pub fn turn_done(&mut self, theme: &Theme) {
        if !self.on() {
            return;
        }
        let e = fx::fade_from_fg(theme.accent2, (900, Interpolation::SineOut));
        self.manager
            .add_unique_effect("turn", e.with_area(self.areas.input));
    }

    pub fn error(&mut self, theme: &Theme) {
        if !self.on() {
            return;
        }
        let e = fx::fade_from(theme.err, theme.err, (900, Interpolation::QuadOut));
        self.manager
            .add_unique_effect("error", e.with_area(self.areas.status));
    }

    pub fn compacted(&mut self, _theme: &Theme) {
        if self.level != MotionLevel::Full {
            return;
        }
        let e = fx::ping_pong(fx::hsl_shift_fg(
            [90.0, 0.0, 0.1],
            (450, Interpolation::SineInOut),
        ));
        self.manager
            .add_unique_effect("compact", e.with_area(self.areas.chat));
    }

    pub fn theme_changed(&mut self, theme: &Theme) {
        if !self.on() {
            return;
        }
        let e = fx::fade_from(theme.bg, theme.bg, (400, Interpolation::QuadOut));
        self.manager
            .add_unique_effect("theme", e.with_area(self.areas.screen));
    }

    /// Run every active effect over the freshly rendered frame.
    pub fn process(&mut self, elapsed: Duration, buf: &mut Buffer, theme: &Theme) {
        if !self.on() {
            return;
        }
        if std::mem::take(&mut self.startup_pending) {
            self.add_startup(theme);
        }
        let area = self.areas.screen;
        self.manager.process_effects(elapsed.into(), buf, area);
    }
}
