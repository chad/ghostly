//! Utopia — gold glass-orb face wrapped in dense flowing wire bands.
//! **PLACEHOLDER.**
//!
//! The avatar reference (`CHARACTERS.utopia`) carries the most elaborate
//! `renderConfig` in the line-up: `glassMode`, fresnel + specular +
//! iridescence + inner glow + a golden `nebulaCloud`, lavender
//! `accentParticles` orbiting the edges, a golden voronoi mesh, and a
//! contour shimmer that morphs gold → lavender. None of that is wired
//! up here yet — this stub builds a Utopia-flavoured character that
//! renders cleanly so the registry / CLI / Eliza wiring can be built
//! before we port the visuals.
//!
//! TODO:
//! - port `utopiaContours()` — ~14 globe-style flowing wire bands +
//!   face-feature contours
//! - port the "shower burst" displace closure
//! - port the glass-orb render config (fresnel, specular, iridescence,
//!   accentParticles, voronoiMesh, contourShimmer, nebulaCloud)

use crate::character::{
    Character, ContourBaseline, Geometry, Palette, RenderConfig, Transition,
};

pub fn build() -> Character {
    Character {
        name: "utopia",
        geometry: utopia_geometry(),
        palette: utopia_palette(),
        // Golden — the wire bands and contour shimmer target lavender
        // in the full implementation; the static colour for now.
        contour_color: [255, 210, 80],
        contour_paths: placeholder_contours(),
        transition: Transition::default(),
        contour_baseline: ContourBaseline {
            outer_width: 2.0,
            inner_width: 1.0,
            outer_alpha: 0.85,
            inner_alpha: 0.95,
            breath_rate: 1.4,
            breath_depth: 0.12,
            ripple_amp: 0.010,
        },
        render_config: RenderConfig::default(),
    }
}

/// Utopia's "diamond face": narrow tall geometry, prominent cheekbones,
/// wide warm eyes. Verbatim from `GEOMETRY_PRESETS.utopia`.
fn utopia_geometry() -> Geometry {
    Geometry {
        head_width: 0.42,
        head_height: 0.60,
        jaw_narrow: 0.55,
        brow_ridge: 0.02,
        eye_socket_depth: 0.05,
        eye_size: 1.4,
        eyeball_bump: 0.11,
        eye_spread: 0.18,
        cheek_bone: 0.22,
        nose_scale: 0.75,
        lip_fullness: 1.1,
        chin_size: 0.025,
        forehead_curve: 0.02,
        temple_indent: 0.06,
        horns: false,
        eye_glow: 1.7,
        aspect_x: 0.80,
        aspect_y: 1.25,
    }
}

fn utopia_palette() -> Palette {
    // Strong gold base with warm-dark shadows + lavender iridescent
    // edge accent (unused until the full render pipeline lands).
    Palette {
        base: [0.95, 0.78, 0.20],
        deep: [0.40, 0.28, 0.08],
        highlight: [1.0, 0.95, 0.50],
        eye: [1.0, 0.92, 0.3],
        eye_rim: None,
        glow: [1.0, 0.85, 0.25],
    }
}

fn placeholder_contours() -> Vec<Vec<[f32; 2]>> {
    // A single subtle brow until the globe wire-band port lands.
    vec![(0..20)
        .map(|j| {
            let t = j as f32 / 19.0;
            [(t - 0.5) * 0.60, 0.30 + (t * std::f32::consts::PI).cos() * 0.035]
        })
        .collect()]
}
