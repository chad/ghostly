//! Eliza — freeq's voice + video agent. **PLACEHOLDER.**
//!
//! Eliza is not in the avatar repo — she lives in
//! `~/src/freeq/freeq-eliza` and currently renders as an SVG cyberpunk
//! orb (`freeq-eliza/src/video.rs`): corner brackets, state sticker,
//! EQ strip, halftone field, glitch transitions, vision PiP, ambient
//! HUD chip. This stub puts her in the ghostly registry so the same
//! particle-face renderer can host her once the SVG → particles port
//! is complete.
//!
//! TODO:
//! - design Eliza's particle geometry — probably a friendly
//!   approachable face (the SVG presence has eyes + a mouth that
//!   lip-syncs to her speech, so the geometry should keep features
//!   readable)
//! - palette — mint-teal `#3effd6` LISTENING accent maps naturally to
//!   a teal/green palette family; could shift on mood
//! - contour — the SVG corner brackets are very Eliza; consider
//!   keeping them as a non-face-following overlay on top of the
//!   particle field
//! - per-mood swap: Eliza already has Idle / Listening / Thinking /
//!   Speaking / Vision moods; each could pick a different palette via
//!   sentiment-style morphing rather than a different `Character`

use crate::character::{
    Character, ContourBaseline, Geometry, Palette, RenderConfig, Transition,
};

pub fn build() -> Character {
    Character {
        name: "eliza",
        geometry: eliza_geometry(),
        palette: eliza_palette(),
        // Mint-teal — matches her LISTENING mood accent in
        // freeq-eliza/src/video.rs.
        contour_color: [62, 255, 214],
        contour_paths: placeholder_contours(),
        transition: Transition::default(),
        contour_baseline: ContourBaseline::default(),
        render_config: RenderConfig::default(),
    }
}

/// A starter geometry — a friendly neutral face with slightly wider
/// eyes (she's listening, not menacing). Tuneable.
fn eliza_geometry() -> Geometry {
    Geometry {
        eye_size: 1.15,
        eye_spread: 0.17,
        cheek_bone: 0.08,
        lip_fullness: 1.05,
        ..Geometry::default()
    }
}

/// Teal palette — extends the [`Palette::GHOST`] template with mint
/// highlights. A first draft; will likely become mood-dependent.
fn eliza_palette() -> Palette {
    Palette {
        base: [0.25, 0.75, 0.65],
        deep: [0.05, 0.18, 0.18],
        highlight: [0.55, 1.0, 0.90],
        eye: [0.4, 1.0, 0.85],
        eye_rim: None,
        glow: [0.25, 0.95, 0.85],
    }
}

fn placeholder_contours() -> Vec<Vec<[f32; 2]>> {
    // Soft brow + a hint of jaw. The SVG presence's bracket frame is
    // probably the more recognizable Eliza signature — port that as a
    // tile overlay rather than a face contour.
    vec![
        (0..18)
            .map(|j| {
                let t = j as f32 / 17.0;
                [(t - 0.5) * 0.54, 0.30 + (t * std::f32::consts::PI).cos() * 0.025]
            })
            .collect(),
        (0..18)
            .map(|j| {
                let t = j as f32 / 17.0;
                let ang = std::f32::consts::PI * 0.18 + t * std::f32::consts::PI * 0.64;
                [ang.cos() * 0.34, -0.12 - ang.sin() * 0.26]
            })
            .collect(),
    ]
}
