//! Narrator — calm strong-blue ghost. **PLACEHOLDER.**
//!
//! The avatar reference (`CHARACTERS.narrator`) has bespoke
//! `defaultContours()` + a "murmuration" displace closure. This stub
//! returns a render-clean character with sensible defaults so the
//! registry has an entry and the binary can list it; full port
//! pending.
//!
//! TODO:
//! - port `defaultContours()` (brow / cheekbones / jaw / nose bridge)
//! - port the murmuration `displace` (swirl in a flock-like pattern)
//! - confirm the `ghost` palette (already in [`Palette::GHOST`])

use crate::character::{
    Character, ContourBaseline, Geometry, Palette, RenderConfig, Transition,
};

pub fn build() -> Character {
    Character {
        name: "narrator",
        geometry: Geometry::default(),
        palette: Palette::GHOST,
        contour_color: [60, 130, 255],
        contour_paths: placeholder_contours(),
        transition: Transition::default(),
        contour_baseline: ContourBaseline {
            outer_width: 3.0,
            inner_width: 1.5,
            outer_alpha: 0.85,
            inner_alpha: 0.95,
            breath_rate: 1.2,
            breath_depth: 0.15,
            ripple_amp: 0.015,
        },
        render_config: RenderConfig::default(),
    }
}

/// Placeholder: a single brow line so the face isn't featureless.
/// Replace with the full `defaultContours()` port from face-character.js.
fn placeholder_contours() -> Vec<Vec<[f32; 2]>> {
    vec![(0..20)
        .map(|j| {
            let t = j as f32 / 19.0;
            [(t - 0.5) * 0.56, 0.32 + (t * std::f32::consts::PI).cos() * 0.02]
        })
        .collect()]
}
