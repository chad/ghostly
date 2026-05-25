//! Eliza — freeq's voice + video agent. Friendly approachable face
//! with mint-teal accents that matches her LISTENING mood colour in
//! the SVG presence. The render gives her wide alert eyes, a soft
//! lip line, a faint teal nebula glow, and a digital-scanline scatter
//! transition that fits the cyberpunk corner-bracket look of her
//! SVG identity in freeq-eliza.
//!
//! She's not in the avatar repo (avatar has narrator/utopia/oblivion
//! only — Eliza is freeq-native), so this is a ghostly-side design
//! rather than a verbatim port. The design rules:
//!   * features must stay readable — eye outlines, mouth, brow — so
//!     a participant can tell they're being looked at
//!   * mood-driven palette via `sentiment::apply_emotion`, not a
//!     separate Character per mood
//!   * no horns / no fire eyes — Eliza is helpful, not ominous

use crate::character::{
    Character, ContourBaseline, Geometry, NebulaCloud, Palette, RenderConfig, Transition,
};

pub fn build() -> Character {
    Character {
        name: "eliza",
        geometry: eliza_geometry(),
        palette: eliza_palette(),
        // Mint-teal — matches the LISTENING mood accent in
        // freeq-eliza/src/video.rs.
        contour_color: [62, 255, 214],
        contour_paths: eliza_contours(),
        transition: scanline_glitch(),
        contour_baseline: ContourBaseline {
            outer_width: 2.8,
            inner_width: 1.4,
            outer_alpha: 0.85,
            inner_alpha: 0.95,
            // Slightly faster breath than narrator — alert / engaged
            // rather than serene.
            breath_rate: 1.5,
            breath_depth: 0.14,
            ripple_amp: 0.012,
        },
        render_config: RenderConfig {
            fresnel_power: 3.0,
            fresnel_intensity: 0.55,
            specular_power: 50.0,
            specular_intensity: 0.6,
            band_glow: false,
            voronoi_mesh: None,
            // Faint teal halo behind the face — reads as her
            // "presence" without dominating the tile.
            nebula_cloud: Some(NebulaCloud {
                color: [40, 220, 200],
                intensity: 0.35,
            }),
            embers: None,
            accent_particles: None,
            vignette: 0.35,
            // When she speaks, particles tint toward bright cyan-mint
            // — same family as her base, just hotter.
            audio_glow: [0.5, 1.0, 0.92],
            audio_glow_strength: 0.55,
        },
    }
}

/// Friendly face: prominent wide eyes (she's listening), softened
/// jaw, gentle cheekbones. No horns. Slightly squashed aspect so the
/// silhouette reads round rather than long, which fits her helpful-
/// assistant identity better than the angular avatar trio.
fn eliza_geometry() -> Geometry {
    Geometry {
        head_width: 0.40,
        head_height: 0.52,
        jaw_narrow: 0.62,
        brow_ridge: 0.025,
        eye_socket_depth: 0.04,
        eye_size: 1.25,
        eyeball_bump: 0.10,
        eye_spread: 0.17,
        cheek_bone: 0.10,
        nose_scale: 0.80,
        lip_fullness: 1.10,
        chin_size: 0.02,
        forehead_curve: 0.03,
        temple_indent: 0.05,
        horns: false,
        eye_glow: 1.25,
        aspect_x: 0.85,
        aspect_y: 1.10,
    }
}

/// Teal palette — extends [`Palette::GHOST`]'s ghost structure with
/// mint highlights. Mood-blended at runtime by
/// [`crate::sentiment::apply_emotion`].
fn eliza_palette() -> Palette {
    Palette {
        base: [0.30, 0.85, 0.72],
        deep: [0.05, 0.18, 0.18],
        highlight: [0.65, 1.0, 0.92],
        eye: [0.45, 1.0, 0.88],
        eye_rim: None,
        glow: [0.30, 0.95, 0.85],
    }
}

/// Facial-feature polylines — brow ridge, eye outlines, nose bridge,
/// mouth, jaw. Drawn over the particle field by `splat_contours`.
fn eliza_contours() -> Vec<Vec<[f32; 2]>> {
    use std::f32::consts::PI;
    let mut out: Vec<Vec<[f32; 2]>> = Vec::with_capacity(6);

    // Brow ridge — softer arch than narrator's, slightly higher.
    out.push(
        (0..20)
            .map(|j| {
                let t = j as f32 / 19.0;
                let x = (t - 0.5) * 0.56;
                [x, 0.30 + (t * PI).cos() * 0.025]
            })
            .collect(),
    );

    // Left eye outline — almond shape, slightly bigger than utopia's
    // for the "alert" read.
    out.push(
        (0..16)
            .map(|j| {
                let t = j as f32 / 15.0;
                let ang = -PI * 0.1 + t * PI * 1.2;
                [-0.17 + ang.cos() * 0.09, 0.20 + ang.sin() * 0.045]
            })
            .collect(),
    );
    // Right eye outline.
    out.push(
        (0..16)
            .map(|j| {
                let t = j as f32 / 15.0;
                let ang = -PI * 0.1 + t * PI * 1.2;
                [0.17 + ang.cos() * 0.09, 0.20 + ang.sin() * 0.045]
            })
            .collect(),
    );

    // Nose bridge — short, vertical.
    out.push(
        (0..8)
            .map(|j| {
                let t = j as f32 / 7.0;
                [0.0, 0.22 - t * 0.28]
            })
            .collect(),
    );

    // Mouth — a faint upturn. Symmetric quadratic dip; the LIP
    // particles below carry most of the mouth read, the contour just
    // hints at the curve so it survives lip-sync motion.
    out.push(
        (0..14)
            .map(|j| {
                let t = j as f32 / 13.0;
                let x = (t - 0.5) * 0.24;
                let y = -0.20 + (0.5 - (t - 0.5).abs()) * 0.06;
                [x, y]
            })
            .collect(),
    );

    // Jaw — softer than narrator's, narrower than utopia's. Same
    // taper trick (narrowest at the midpoint, opening at the cheeks
    // and chin) for a recognisable silhouette.
    out.push(
        (0..24)
            .map(|j| {
                let t = j as f32 / 23.0;
                let ang = PI * 0.18 + t * PI * 0.64;
                let taper = if t > 0.5 { (1.0 - t) * 2.0 } else { t * 2.0 };
                let jw = 0.34 - t * 0.06 * taper;
                [ang.cos() * jw, -0.14 - ang.sin() * 0.28]
            })
            .collect(),
    );

    out
}

/// **Scanline glitch** — the scatter transition. Particles drift on
/// horizontal scan lines (Y bucketed into bands) with each band
/// shifting in a different direction by an amount that decays over
/// the transition. Reads as a CRT glitch / VHS tear, which fits
/// Eliza's cyberpunk corner-bracket SVG identity. Z is randomised
/// per-particle to give the glitch some depth and avoid a flat plane
/// scatter.
fn scanline_glitch() -> Transition {
    Transition {
        displace: Box::new(|_fnx, fny, ease, time, seed| {
            // 16 horizontal scan bands across the face. Each band has
            // its own phase, so adjacent rows shear in different
            // directions.
            let band = (fny * 8.0).floor();
            let band_phase = band * 1.37 + (time * 0.3).sin();
            // ±0.6 horizontal shear, scaled by ease.
            let dx = (band_phase + seed * 6.28).sin() * ease * 0.6;
            // Slight vertical jitter — keeps the scanlines from
            // looking too perfectly horizontal.
            let dy = (seed - 0.5) * ease * 0.15;
            let dz = (seed - 0.5) * ease * 0.4;
            [dx, dy, dz]
        }),
        enter_speed: 2.2,
        exit_speed: 2.8,
    }
}
