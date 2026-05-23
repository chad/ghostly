//! Narrator — calm strong-blue ghost. The avatar's default presence:
//! soft brow ridge, gentle cheekbones, anchored jaw line, a vertical
//! nose bridge. Particles murmurate when scattered — a flock-like
//! swirl rather than a violent displacement.
//!
//! Full port of `CHARACTERS.narrator` in face-character.js +
//! `defaultContours()` in the same file.

use crate::character::{
    Character, ContourBaseline, Geometry, Palette, RenderConfig, Transition,
};

pub fn build() -> Character {
    Character {
        name: "narrator",
        geometry: Geometry::default(),
        palette: Palette::GHOST,
        contour_color: [60, 130, 255],
        contour_paths: default_contours(),
        transition: murmuration(),
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

/// Standard facial-feature polylines used by every "default" face: a
/// soft brow, symmetric cheekbones, a curved jaw, and a vertical nose
/// bridge. Verbatim from `defaultContours()` in face-character.js.
fn default_contours() -> Vec<Vec<[f32; 2]>> {
    let mut out: Vec<Vec<[f32; 2]>> = Vec::with_capacity(5);
    use std::f32::consts::PI;

    // Brow ridge — gentle arch across the upper face.
    out.push((0..20).map(|j| {
        let t = j as f32 / 19.0;
        let x = (t - 0.5) * 0.56;
        [x, 0.32 + (t * PI).cos() * 0.02]
    }).collect());

    // Left cheekbone — diagonal sweep down + in.
    out.push((0..12).map(|j| {
        let t = j as f32 / 11.0;
        [-0.18 - t * 0.14, 0.15 - t * 0.2]
    }).collect());

    // Right cheekbone — mirror of the left.
    out.push((0..12).map(|j| {
        let t = j as f32 / 11.0;
        [0.18 + t * 0.14, 0.15 - t * 0.2]
    }).collect());

    // Jaw line — arcs from the lower cheek through the chin and back.
    // The taper factor narrows the jaw most at its midpoint, giving
    // it a soft chiseled look rather than a hard angle.
    out.push((0..24).map(|j| {
        let t = j as f32 / 23.0;
        let ang = PI * 0.15 + t * PI * 0.7;
        let taper = if t > 0.5 { (1.0 - t) * 2.0 } else { t * 2.0 };
        let jw = 0.35 - t * 0.1 * taper;
        [ang.cos() * jw, -0.15 - ang.sin() * 0.32]
    }).collect());

    // Nose bridge — vertical line from the brow down to the upper lip.
    out.push((0..8).map(|j| {
        let t = j as f32 / 7.0;
        [0.0, 0.28 - t * 0.35]
    }).collect());

    out
}

/// **Murmuration** — particles swirl in a flock-like pattern when
/// scattered. Each particle picks an angular phase from its seed, then
/// orbits at a per-particle radius modulated by both seed and time.
/// The Y component is gently squashed (`0.6`) so the swirl feels more
/// horizontal — like a flock of birds banking — than spherical.
///
/// Direct port of `CHARACTERS.narrator.transition.displace` in
/// face-character.js.
fn murmuration() -> Transition {
    Transition {
        displace: Box::new(|_fnx, _fny, ease, time, seed| {
            use std::f32::consts::TAU;
            let phase = seed * TAU + time * 1.5;
            let radius = ease * (2.0 + seed * 3.0);
            let dx = phase.cos() * radius * (seed * 7.0 + time * 0.3).cos();
            let dy = phase.sin() * radius * 0.6;
            let dz = (phase * 0.7 + seed * 5.0).sin() * ease * 2.0;
            [dx, dy, dz]
        }),
        enter_speed: 1.8,
        exit_speed: 2.5,
    }
}
