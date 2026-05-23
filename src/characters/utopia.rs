//! Utopia — gold glass-orb face wrapped in dense flowing wire bands.
//!
//! Full port of `CHARACTERS.utopia` + `utopiaContours()` +
//! `GEOMETRY_PRESETS.utopia` + `PALETTES.utopia` in face-character.js
//! /face-gen.js. The glass-orb render config is partly wired into the
//! software renderer (nebula cloud, accent particles, band glow,
//! voronoi mesh); the GPU-only knobs (`fresnel_power`,
//! `specular_power`, iridescence) are carried verbatim for a future
//! wgpu backend.

use crate::character::{
    AccentParticles, Character, ContourBaseline, Geometry, NebulaCloud, Palette, RenderConfig,
    Transition, VoronoiMesh,
};

pub fn build() -> Character {
    Character {
        name: "utopia",
        geometry: utopia_geometry(),
        palette: utopia_palette(),
        // Golden wire bands — picks up lavender accents at the edges
        // via the iridescence path on GPU; static gold here.
        contour_color: [255, 210, 80],
        contour_paths: utopia_contours(),
        transition: shower_burst(),
        contour_baseline: ContourBaseline {
            outer_width: 2.0,      // thinner outer glow — wire-band look
            inner_width: 1.0,      // crisp inner line
            outer_alpha: 0.85,
            inner_alpha: 0.95,
            breath_rate: 1.4,
            breath_depth: 0.12,
            ripple_amp: 0.010,     // smoother glass bands — less ripple
        },
        render_config: RenderConfig {
            fresnel_power: 4.0,
            fresnel_intensity: 0.4,
            specular_power: 60.0,
            specular_intensity: 0.7,
            band_glow: true,
            // Faint golden cracked-glass overlay completes the
            // "crystal orb" feel.
            voronoi_mesh: Some(VoronoiMesh {
                color: [255, 200, 60],
                alpha: 0.12,
                line_width: 0.5,
            }),
            // Dim golden glow behind the orb — bathes the room.
            nebula_cloud: Some(NebulaCloud {
                color: [255, 200, 40],
                intensity: 0.5,
            }),
            // Floating lavender motes around the edges.
            accent_particles: Some(AccentParticles {
                count: 250,
                color: [180, 150, 255],
                alpha: 0.35,
                radius: 2.8,
            }),
        },
    }
}

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
    Palette {
        base: [0.95, 0.78, 0.20],
        deep: [0.40, 0.28, 0.08],
        highlight: [1.0, 0.95, 0.50],
        eye: [1.0, 0.92, 0.3],
        eye_rim: None,
        glow: [1.0, 0.85, 0.25],
    }
}

/// The signature Utopia look: dense horizontal wire bands wrapping the
/// face like a glass globe, plus standard facial feature contours
/// layered on top. Verbatim port of `utopiaContours()` in
/// face-character.js.
fn utopia_contours() -> Vec<Vec<[f32; 2]>> {
    use std::f32::consts::PI;
    let mut out: Vec<Vec<[f32; 2]>> = Vec::new();

    // ── ~14 globe-style flowing wire bands ──
    // Each band: a polyline at a fixed Y, x ranging across the face
    // width at that latitude, with multi-frequency wave displacement.
    const BAND_COUNT: usize = 14;
    const PTS: usize = 32;
    for b in 0..BAND_COUNT {
        let band_y = -0.35 + (b as f32 / (BAND_COUNT as f32 - 1.0)) * 0.85;
        // Width follows an ellipse — narrower at top + chin, wider at
        // the cheek midline. Sqrt is the trick that gives the wire
        // bands their bulging "glass" silhouette.
        let face_width_at_y =
            0.42 * (1.0 - ((band_y - 0.05) / 0.55).powi(2)).max(0.05).sqrt();
        let phase = b as f32 * 0.73 + (b as f32).powi(2) * 0.12;
        let band: Vec<[f32; 2]> = (0..PTS)
            .map(|j| {
                let t = j as f32 / (PTS as f32 - 1.0);
                let x = (t - 0.5) * 2.0 * face_width_at_y;
                // Multi-frequency wave — gives the band a liquid flow
                // shape rather than a regular sine.
                let wave = (t * PI * 2.5 + phase).sin() * 0.015
                    + (t * PI * 5.3 + phase * 1.7).sin() * 0.008
                    + (t * PI * 1.2 + phase * 0.5).cos() * 0.012;
                [x, band_y + wave]
            })
            .collect();
        out.push(band);
    }

    // ── Facial-feature contours layered on top of the bands ──
    // Soft brow — gentle arch.
    out.push((0..20).map(|j| {
        let t = j as f32 / 19.0;
        let x = (t - 0.5) * 0.60;
        [x, 0.30 + (t * PI).cos() * 0.035]
    }).collect());

    // Nose bridge.
    out.push((0..8).map(|j| {
        let t = j as f32 / 7.0;
        [0.0, 0.26 - t * 0.30]
    }).collect());

    // Left eye contour — open almond shape (Utopia's wide warm eyes).
    out.push((0..14).map(|j| {
        let t = j as f32 / 13.0;
        let ang = -PI * 0.1 + t * PI * 1.2;
        [-0.17 + ang.cos() * 0.08, 0.22 + ang.sin() * 0.04]
    }).collect());

    // Right eye contour.
    out.push((0..14).map(|j| {
        let t = j as f32 / 13.0;
        let ang = -PI * 0.1 + t * PI * 1.2;
        [0.17 + ang.cos() * 0.08, 0.22 + ang.sin() * 0.04]
    }).collect());

    // Soft jaw — gentle arc.
    out.push((0..24).map(|j| {
        let t = j as f32 / 23.0;
        let ang = PI * 0.18 + t * PI * 0.64;
        let taper = if t > 0.5 { (1.0 - t) * 2.0 } else { t * 2.0 };
        let jw = 0.38 - t * 0.05 * taper;
        [ang.cos() * jw, -0.12 - ang.sin() * 0.28]
    }).collect());

    // Mouth — barely upturned, abstract rather than smiley.
    out.push((0..12).map(|j| {
        let t = j as f32 / 11.0;
        let x = (t - 0.5) * 0.20;
        let y = -0.19 + (t - 0.5).powi(2) * 0.04;
        [x, y]
    }).collect());

    out
}

/// **Shower burst** — particles explode outward radially from the
/// face centre when scattered, with an extra upward drift weighted by
/// seed. Direct port of `CHARACTERS.utopia.transition.displace`.
fn shower_burst() -> Transition {
    Transition {
        displace: Box::new(|fnx, fny, ease, _time, seed| {
            let ang = (fny - 0.1).atan2(fnx) + seed * 0.5;
            let dist = ease * (3.0 + seed * 5.0);
            [
                ang.cos() * dist,
                ang.sin() * dist + ease * seed * 2.0,
                (seed - 0.5) * ease * 4.0,
            ]
        }),
        enter_speed: 1.8,
        exit_speed: 2.5,
    }
}
