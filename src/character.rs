//! Character — a self-contained bundle that fully describes a face's
//! look and motion.
//!
//! Ported from `~/src/avatar/static/viz/face-character.js`. Each
//! character carries everything the renderer needs:
//!
//! - [`Geometry`] — head/eye/jaw/horn proportions that drive the depth
//!   map sampled by [`crate::face::generate_face`].
//! - [`Palette`] — base/deep/highlight/eye colours.
//! - [`Contour`] — 2D facial-feature polylines drawn as glowing strokes
//!   on top of the particle field.
//! - [`Transition`] — how particles displace when scattered, plus how
//!   fast they re-converge.
//! - [`ContourBaseline`] — line widths + breathing pulse for the silent
//!   "alive" state.
//! - [`RenderConfig`] — fresnel / specular / nebula / wireframe knobs.
//!
//! The avatar JS object had ad-hoc fields per character; here every
//! field exists for every character (some Default) so the renderer
//! never special-cases by name.

/// Procedural face proportions. Defaults reproduce the avatar's
/// "neutral" geometry preset. Set fields that differ per character and
/// leave the rest at the avatar defaults.
///
/// Mirrors `GEOMETRY_PRESETS` in `face-gen.js`.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub head_width: f32,
    pub head_height: f32,
    pub jaw_narrow: f32,
    pub brow_ridge: f32,
    pub eye_socket_depth: f32,
    pub eye_size: f32,
    pub eyeball_bump: f32,
    pub eye_spread: f32,
    pub cheek_bone: f32,
    pub nose_scale: f32,
    pub lip_fullness: f32,
    pub chin_size: f32,
    pub forehead_curve: f32,
    pub temple_indent: f32,
    pub horns: bool,
    pub eye_glow: f32,
    pub aspect_x: f32,
    pub aspect_y: f32,
}

impl Default for Geometry {
    fn default() -> Self {
        // Avatar's "default" preset — all fields at their fall-back
        // values in faceDepth().
        Self {
            head_width: 0.42,
            head_height: 0.55,
            jaw_narrow: 0.4,
            brow_ridge: 0.07,
            eye_socket_depth: 0.13,
            eye_size: 1.0,
            eyeball_bump: 0.05,
            eye_spread: 0.16,
            cheek_bone: 0.06,
            nose_scale: 1.0,
            lip_fullness: 1.0,
            chin_size: 0.06,
            forehead_curve: 0.04,
            temple_indent: 0.03,
            horns: false,
            eye_glow: 1.0,
            aspect_x: 1.0,
            aspect_y: 1.0,
        }
    }
}

/// Five-channel colour palette. Mirrors the JS `PALETTES` entries in
/// `face-gen.js`. RGB components are `0.0..=1.0`.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    pub base: [f32; 3],
    pub deep: [f32; 3],
    pub highlight: [f32; 3],
    pub eye: [f32; 3],
    /// Outer rim colour for "fire eye" mode (Oblivion). `None` =
    /// classic single-tone eye colour.
    pub eye_rim: Option<[f32; 3]>,
    pub glow: [f32; 3],
}

impl Palette {
    /// "Ghost" palette — Narrator's strong-blue look. The avatar uses
    /// this as the universal fallback.
    pub const GHOST: Self = Self {
        base: [0.35, 0.50, 0.90],
        deep: [0.05, 0.08, 0.25],
        highlight: [0.65, 0.80, 1.0],
        eye: [0.4, 0.7, 1.0],
        eye_rim: None,
        glow: [0.2, 0.4, 1.0],
    };
}

/// A 2D facial-feature polyline. Coordinates are in the same normalized
/// face space as [`Geometry`] (`x ∈ [-0.5, 0.5]`, `y ∈ [-0.65, 0.85]`).
pub type Contour = Vec<[f32; 2]>;

/// How a particle displaces when scattered. The renderer calls this
/// with the particle's normalized target position + a per-particle seed
/// and a `0..=1` ease, and gets back a 3D offset that's added to the
/// scattered position.
///
/// Type-erased (`Box<dyn Fn>`) so a [`Character`] can be a plain value
/// rather than a generic parameter — keeps the registry simple.
pub type Displace =
    Box<dyn Fn(/* fnx */ f32, /* fny */ f32, /* ease */ f32, /* time */ f32, /* seed */ f32) -> [f32; 3] + Send + Sync>;

/// Particle transition behaviour. `enter_speed` / `exit_speed` are the
/// per-particle interpolation rates toward `target` / `scattered`
/// positions when materializing or dissolving the face.
pub struct Transition {
    pub displace: Displace,
    pub enter_speed: f32,
    pub exit_speed: f32,
}

impl Default for Transition {
    fn default() -> Self {
        // A neutral fall-back: pure radial drift. Placeholder characters
        // use this until they get a real transition.
        Self {
            displace: Box::new(|fnx, fny, ease, _time, seed| {
                let ang = (fny - 0.1).atan2(fnx) + seed * 0.5;
                let dist = ease * (2.0 + seed * 2.0);
                [
                    ang.cos() * dist,
                    ang.sin() * dist,
                    (seed - 0.5) * ease * 2.0,
                ]
            }),
            enter_speed: 1.8,
            exit_speed: 2.5,
        }
    }
}

/// Stroke + breathing parameters for the contour overlay when the face
/// is silent. Mirrors `contourBaseline` in the JS spec.
#[derive(Clone, Copy, Debug)]
pub struct ContourBaseline {
    pub outer_width: f32,
    pub inner_width: f32,
    pub outer_alpha: f32,
    pub inner_alpha: f32,
    pub breath_rate: f32,
    pub breath_depth: f32,
    pub ripple_amp: f32,
}

impl Default for ContourBaseline {
    fn default() -> Self {
        Self {
            outer_width: 3.0,
            inner_width: 1.5,
            outer_alpha: 0.85,
            inner_alpha: 0.95,
            breath_rate: 1.2,
            breath_depth: 0.15,
            ripple_amp: 0.015,
        }
    }
}

/// Fresnel / specular / mesh-overlay knobs that mostly affect the
/// post-pass on a real GPU pipeline. We carry the values verbatim so a
/// future wgpu backend can read them, and the software backend uses a
/// subset (`fresnel_intensity` adds rim glow; `voronoi_mesh` if `Some`
/// draws a wireframe overlay).
#[derive(Clone, Debug)]
pub struct RenderConfig {
    pub fresnel_power: f32,
    pub fresnel_intensity: f32,
    pub specular_power: f32,
    pub specular_intensity: f32,
    pub voronoi_mesh: Option<VoronoiMesh>,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            fresnel_power: 5.0,
            fresnel_intensity: 0.5,
            specular_power: 40.0,
            specular_intensity: 1.0,
            voronoi_mesh: None,
        }
    }
}

/// Faint cracked-glass wireframe overlay (`voronoiMesh` in the JS
/// spec). Oblivion uses a red-tinted mesh; Utopia would use gold.
#[derive(Clone, Copy, Debug)]
pub struct VoronoiMesh {
    pub color: [u8; 3],
    pub alpha: f32,
    pub line_width: f32,
}

/// A complete character — name + everything the renderer needs.
///
/// Built once and reused for the life of a session. Cheap to read,
/// `!Clone` only because [`Transition::displace`] is a boxed closure;
/// if you need to clone you can rebuild from the matching
/// `Characters::…` factory function.
pub struct Character {
    pub name: &'static str,
    pub geometry: Geometry,
    pub palette: Palette,
    /// 0-255 RGB for the contour strokes (independent of the particle
    /// palette — Oblivion's contour is a different red than its base).
    pub contour_color: [u8; 3],
    /// Polylines in normalized face space. Boxed so a Character can be
    /// rebuilt cheaply.
    pub contour_paths: Vec<Contour>,
    pub transition: Transition,
    pub contour_baseline: ContourBaseline,
    pub render_config: RenderConfig,
}
