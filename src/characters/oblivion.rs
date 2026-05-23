//! Oblivion — predatory horned face, blood-red palette, fire eyes.
//!
//! Full port of the JS character spec in
//! `~/src/avatar/static/viz/face-character.js` (`CHARACTERS.oblivion`)
//! plus `GEOMETRY_PRESETS.oblivion`, `PALETTES.oblivion`, and
//! `oblivionContours()` from `face-gen.js`. Bit-faithful to the
//! original so a side-by-side compare reads as the same face.

use crate::character::{
    Character, ContourBaseline, Geometry, Palette, RenderConfig, Transition, VoronoiMesh,
};

pub fn build() -> Character {
    Character {
        name: "oblivion",
        geometry: geometry(),
        palette: palette(),
        // Saturated red — the contour pops off the dark base palette
        // rather than blending in.
        contour_color: [255, 40, 20],
        contour_paths: contours(),
        transition: transition(),
        contour_baseline: ContourBaseline {
            outer_width: 3.5,
            inner_width: 2.0,
            outer_alpha: 0.85,
            inner_alpha: 0.95,
            // Slower, heavier breathing — menace.
            breath_rate: 0.8,
            breath_depth: 0.20,
            ripple_amp: 0.018,
        },
        render_config: RenderConfig {
            // Tighter fresnel falloff so eyes + horns still read.
            fresnel_power: 3.5,
            fresnel_intensity: 0.35,
            // Gentler specular — less blinding.
            specular_power: 40.0,
            specular_intensity: 0.6,
            // Faint cracked-glass red wireframe — completes the
            // "fractured idol" look.
            voronoi_mesh: Some(VoronoiMesh {
                color: [255, 60, 30],
                alpha: 0.14,
                line_width: 0.5,
            }),
            // No nebula / accent ring — Oblivion's menace comes from
            // sharp lines, not soft luminance.
            nebula_cloud: None,
            accent_particles: None,
            band_glow: false,
            // Heavy vignette — drops the corners deep into black so
            // the predator face spotlights center-screen. Matches
            // avatar "rage" / "inferno" range (0.7-0.75).
            vignette: 0.72,
            // Audio shifts particles toward orange-fire — same hot
            // colour as the eyes. The face *combusts* when she speaks.
            audio_glow: [1.0, 0.45, 0.0],
            audio_glow_strength: 0.85,
        },
    }
}

fn geometry() -> Geometry {
    Geometry {
        head_width: 0.39,        // narrow skull
        head_height: 0.60,       // elongated
        jaw_narrow: 0.58,        // razor jaw
        brow_ridge: 0.16,        // massive overhanging brow
        eye_socket_depth: 0.14,  // deep but not invisible — fire must show
        eye_size: 0.75,          // narrow mean slits
        eyeball_bump: 0.08,      // eyeballs proud so they glow
        eye_spread: 0.14,        // closer together (predatory focus)
        cheek_bone: 0.12,        // sharp + gaunt
        nose_scale: 1.2,         // large hooked nose
        lip_fullness: 0.4,       // thin cruel line
        chin_size: 0.10,         // strong jutting chin
        forehead_curve: 0.015,   // flat heavy forehead
        temple_indent: 0.06,     // deeply gaunt
        horns: true,
        eye_glow: 2.5,           // inferno
        aspect_x: 1.0,
        aspect_y: 1.0,
    }
}

fn palette() -> Palette {
    Palette {
        base: [0.75, 0.08, 0.08],
        deep: [0.20, 0.02, 0.02],
        highlight: [1.0, 0.20, 0.12],
        eye: [1.0, 0.45, 0.0],          // orange-yellow fire core
        eye_rim: Some([1.0, 0.1, 0.0]), // red outer rim — drives fire-eye blend
        glow: [1.0, 0.15, 0.05],
    }
}

/// Sharper jaw, deeper brow, horn ridges. Matches `oblivionContours()`
/// in face-character.js.
fn contours() -> Vec<Vec<[f32; 2]>> {
    let mut out: Vec<Vec<[f32; 2]>> = Vec::with_capacity(7);

    // Heavy brow ridge — thicker, angular.
    out.push((0..20).map(|j| {
        let t = j as f32 / 19.0;
        let x = (t - 0.5) * 0.52;
        let y = 0.36 + (t * std::f32::consts::PI).cos() * 0.04 - (t - 0.5).abs() * 0.03;
        [x, y]
    }).collect());

    // Left cheekbone — sharp angle.
    out.push((0..14).map(|j| {
        let t = j as f32 / 13.0;
        [-0.16 - t * 0.18, 0.18 - t * 0.25]
    }).collect());

    // Right cheekbone.
    out.push((0..14).map(|j| {
        let t = j as f32 / 13.0;
        [0.16 + t * 0.18, 0.18 - t * 0.25]
    }).collect());

    // Razor jaw — narrower than default.
    out.push((0..24).map(|j| {
        let t = j as f32 / 23.0;
        let ang = std::f32::consts::PI * 0.1 + t * std::f32::consts::PI * 0.8;
        let taper = if t > 0.5 { (1.0 - t) * 2.0 } else { t * 2.0 };
        let jw = 0.28 - t * 0.08 * taper;
        [ang.cos() * jw, -0.18 - ang.sin() * 0.38]
    }).collect());

    // Nose bridge — longer, sharper.
    out.push((0..10).map(|j| {
        let t = j as f32 / 9.0;
        [0.0, 0.32 - t * 0.42]
    }).collect());

    // Left horn ridge.
    out.push((0..12).map(|j| {
        let t = j as f32 / 11.0;
        let x = -0.22 - t * 0.12 - (t * std::f32::consts::PI * 0.7).sin() * 0.06;
        let y = 0.38 + t * 0.35 + (t * std::f32::consts::PI).sin() * 0.08;
        [x, y]
    }).collect());

    // Right horn ridge.
    out.push((0..12).map(|j| {
        let t = j as f32 / 11.0;
        let x = 0.22 + t * 0.12 + (t * std::f32::consts::PI * 0.7).sin() * 0.06;
        let y = 0.38 + t * 0.35 + (t * std::f32::consts::PI).sin() * 0.08;
        [x, y]
    }).collect());

    out
}

/// "Ominous sink" — particles compress inward and drift down, as if
/// gravity were pulling the face into the floor. Matches the JS
/// `displace` for oblivion.
fn transition() -> Transition {
    Transition {
        displace: Box::new(|fnx, _fny, ease, _time, seed| {
            let dx = -fnx * ease * 0.4 * 3.5;          // compress
            let dy = -ease * 2.5 * (0.5 + seed);       // sink
            let dz = ease * (seed - 0.5) * 1.5;
            [dx, dy, dz]
        }),
        enter_speed: 1.8,
        exit_speed: 2.5,
    }
}
