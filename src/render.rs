//! Render a face for one frame.
//!
//! [`Renderer`] is a soft pipeline: software-rasterizes a particle
//! field + contours into a `tiny_skia::Pixmap`. Cheap enough to run at
//! 30fps on a single core at 640×360 with ~12K particles; pluggable
//! into a video encoder downstream (Eliza's MoQ path uses the same
//! Pixmap shape).
//!
//! Animation state lives in [`FaceState`] — separated from the
//! immutable [`Character`] so a single character can drive many tiles
//! and a single tile can swap characters without rebuilding particles.

use tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke, Transform};

use crate::character::{Character, ContourBaseline};
use crate::face::{generate_face, Particle};

/// Per-tile mutable state. Holds the particle field plus the cached
/// animation clock — separate from [`Character`] so the same character
/// can be rendered on N independent tiles.
pub struct FaceState {
    pub character_name: &'static str,
    particles: Vec<Particle>,
    /// Live position per particle (eases from `scattered` toward
    /// `target` as `materialized` rises). Same length as `particles`.
    live: Vec<[f32; 3]>,
    /// Where each particle goes when scattered (set by the transition).
    scattered: Vec<[f32; 3]>,
    /// `0..=1` materialization per particle.
    pt: Vec<f32>,
    /// Global `0..=1`: 0 = fully scattered, 1 = fully on the face.
    pub materialized: f32,
}

impl FaceState {
    /// Build a fresh particle field for `character`. `particle_count`
    /// is total field size — about 60% of the avatar's count looks
    /// great at 640×360.
    pub fn new(character: &Character, particle_count: usize, scale: f32, seed: u64) -> Self {
        let particles = generate_face(particle_count, scale, &character.geometry, &character.palette, seed);
        let live: Vec<[f32; 3]> = particles.iter().map(|p| p.target).collect();
        // Initial scatter positions — replaced as soon as the
        // transition fires; reasonable default in case the renderer is
        // asked for a frame before scattering.
        let scattered: Vec<[f32; 3]> = particles.iter().map(|p| p.target).collect();
        let pt: Vec<f32> = vec![1.0; particles.len()]; // start materialized
        Self {
            character_name: character.name,
            particles,
            live,
            scattered,
            pt,
            materialized: 1.0,
        }
    }

    /// Scatter — fire the character's transition. Each particle gets a
    /// new scatter destination via [`Character::transition`].
    pub fn scatter(&mut self, character: &Character, time: f32) {
        for (i, p) in self.particles.iter().enumerate() {
            let fnx = p.target[0];
            let fny = p.target[1];
            let off = (character.transition.displace)(fnx, fny, 1.0, time, p.seed);
            self.scattered[i] = [p.target[0] + off[0], p.target[1] + off[1], p.target[2] + off[2]];
        }
        self.materialized = 0.0;
    }

    /// Advance the per-particle materialization toward `target`
    /// (`1.0` = on the face, `0.0` = fully scattered).
    pub fn step(&mut self, character: &Character, target: f32, dt: f32) {
        self.materialized += (target - self.materialized) * (dt * 1.5).min(1.0);
        for i in 0..self.particles.len() {
            let speed = if target > self.pt[i] {
                character.transition.enter_speed
            } else {
                character.transition.exit_speed
            };
            let raw = (target - self.pt[i]) * (dt * speed).min(1.0);
            self.pt[i] += raw;

            // Per-particle interpolation between scattered and target.
            let p = self.pt[i].clamp(0.0, 1.0);
            let t = &self.particles[i].target;
            let s = &self.scattered[i];
            self.live[i] = [
                s[0] * (1.0 - p) + t[0] * p,
                s[1] * (1.0 - p) + t[1] * p,
                s[2] * (1.0 - p) + t[2] * p,
            ];
        }
    }
}

/// Settings the [`Renderer`] needs but a [`Character`] shouldn't carry
/// (per-tile concerns: dimensions, particle count, breathing clock).
#[derive(Clone, Copy, Debug)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    /// Per-frame "trail" fill — higher = faster clear, lower = longer
    /// motion smear. Avatar's default range is 0.10..0.35.
    pub trail: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 640,
            height: 360,
            trail: 0.22,
        }
    }
}

/// Stateful renderer — owns the Pixmap so the trail can carry between
/// frames (the previous frame's darkened content is what makes the
/// particles read as motion).
pub struct Renderer {
    pixmap: Pixmap,
    settings: RenderSettings,
}

impl Renderer {
    pub fn new(settings: RenderSettings) -> Option<Self> {
        let mut pixmap = Pixmap::new(settings.width, settings.height)?;
        // Start opaque black so the per-frame fade has a stable
        // baseline and the PNG output isn't transparent where no
        // particles paint. `tiny_skia` uses premultiplied RGBA — `(0,
        // 0, 0, 255)` is "opaque black".
        for px in pixmap.data_mut().chunks_exact_mut(4) {
            px[3] = 255;
        }
        Some(Self { pixmap, settings })
    }

    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Render one frame. `time` is monotonic seconds since renderer
    /// start — drives the breathing pulse + idle jitter.
    pub fn render(&mut self, character: &Character, state: &FaceState, time: f32) -> &Pixmap {
        let w = self.settings.width as f32;
        let h = self.settings.height as f32;

        // Trail / clear. Drop the previous frame toward black by the
        // trail alpha — gives motion smear when particles move.
        self.fade_to_black(self.settings.trail);

        // Map normalized particle space to pixels. Center the face;
        // `sc` is the smaller dimension × 0.38 (matches face-of-god).
        let cx = w * 0.5;
        let cy = h * 0.5;
        let sc = w.min(h) * 0.38;

        let breath =
            (time * character.contour_baseline.breath_rate * std::f32::consts::TAU).sin() * 0.5
                + 0.5;
        let breath_pulse = 1.0 + (breath - 0.5) * character.contour_baseline.breath_depth;

        // ── Particle pass ─────────────────────────────────────────
        let data = self.pixmap.data_mut();
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        for (i, p) in state.particles.iter().enumerate() {
            let pos = state.live[i];

            // Idle drift — tiny per-particle wobble so a fully-
            // materialized face still moves. Cheap, deterministic.
            let nt = time * 0.25;
            let idx = i as f32;
            let drift_x = (idx * 0.00097 + nt).sin()
                * (idx * 0.00071 + nt * 0.7).cos()
                * 0.025;
            let drift_y = (idx * 0.00127 + nt * 1.1).cos()
                * (idx * 0.00089 + nt * 0.5).sin()
                * 0.020;
            let drift_z = (idx * 0.00167 + nt * 0.8).sin() * 0.015;

            let x = pos[0] + drift_x;
            let y = pos[1] + drift_y;
            let z = pos[2] + drift_z;

            // Perspective project. Slightly squashed so particles in
            // front read bigger than ones behind.
            let depth = 1.0 / (2.5 + z);
            let sx = cx + x * sc * depth;
            // Canvas Y is inverted (top-left origin) — flip.
            let sy = cy - y * sc * depth;

            // Size grows with materialization + breath.
            let size = (1.0 + state.pt[i] * 1.6) * depth * breath_pulse;
            let half = size * 0.5;
            let x0 = (sx - half).floor() as i32;
            let y0 = (sy - half).floor() as i32;
            let x1 = (sx + half).ceil() as i32;
            let y1 = (sy + half).ceil() as i32;

            let r = (p.color[0] * 255.0) as u32;
            let g = (p.color[1] * 255.0) as u32;
            let b = (p.color[2] * 255.0) as u32;
            // Alpha climbs with materialization + per-particle base.
            let a_f = p.color[3] * (0.5 + state.pt[i] * 0.5);
            if a_f < 0.01 {
                continue;
            }
            let a = (a_f * 255.0).min(255.0) as u32;

            for py in y0..=y1 {
                if py < 0 || py >= ph as i32 {
                    continue;
                }
                for px in x0..=x1 {
                    if px < 0 || px >= pw as i32 {
                        continue;
                    }
                    let off = ((py as usize) * pw + (px as usize)) * 4;
                    // Premultiplied additive-screen blend onto BGRA.
                    blend_add_premul(&mut data[off..off + 4], r as u8, g as u8, b as u8, a as u8);
                }
            }
        }

        // ── Contour pass ──────────────────────────────────────────
        // Strokes layered on top — outer glow under, sharper inner line over.
        self.stroke_contours(character, breath, cx, cy, sc);

        // ── Voronoi-mesh overlay (Oblivion uses a red one) ───────
        if let Some(mesh) = character.render_config.voronoi_mesh {
            self.draw_voronoi_overlay(mesh, time);
        }

        &self.pixmap
    }

    /// Fade RGB toward black by `alpha` (0..1) while keeping the pixmap
    /// opaque. Cheap — a straight multiplication in the RGBA buffer.
    /// Alpha is held at 255 so PNG output stays opaque and the trail
    /// effect behaves like the JS Canvas version (a black fillRect
    /// every frame).
    fn fade_to_black(&mut self, alpha: f32) {
        let factor = (1.0 - alpha.clamp(0.0, 1.0)) * 255.0;
        let inv = factor as u32;
        let data = self.pixmap.data_mut();
        for px in data.chunks_exact_mut(4) {
            px[0] = ((px[0] as u32 * inv) / 255) as u8;
            px[1] = ((px[1] as u32 * inv) / 255) as u8;
            px[2] = ((px[2] as u32 * inv) / 255) as u8;
            px[3] = 255;
        }
    }

    /// Draw all contour polylines using tiny_skia's stroker. Two-pass:
    /// a fat outer-glow stroke first, a crisp inner line on top.
    fn stroke_contours(&mut self, character: &Character, breath: f32, cx: f32, cy: f32, sc: f32) {
        let cb: ContourBaseline = character.contour_baseline;
        let outer_w = cb.outer_width * (1.0 + (breath - 0.5) * cb.breath_depth);
        let inner_w = cb.inner_width * (1.0 + (breath - 0.5) * cb.breath_depth);
        let outer_a = (cb.outer_alpha * 255.0) as u8;
        let inner_a = (cb.inner_alpha * 255.0) as u8;

        for path in &character.contour_paths {
            if path.len() < 2 {
                continue;
            }
            // Build a tiny_skia path from the normalized polyline.
            let mut pb = PathBuilder::new();
            for (j, p) in path.iter().enumerate() {
                let sx = cx + p[0] * sc;
                let sy = cy - p[1] * sc;
                if j == 0 {
                    pb.move_to(sx, sy);
                } else {
                    pb.line_to(sx, sy);
                }
            }
            let Some(built) = pb.finish() else { continue };

            // Outer glow — softer + wider.
            let mut paint = Paint::default();
            paint.set_color(Color::from_rgba8(
                character.contour_color[0],
                character.contour_color[1],
                character.contour_color[2],
                outer_a / 2,
            ));
            paint.anti_alias = true;
            let mut stroke = Stroke::default();
            stroke.width = outer_w;
            stroke.line_cap = tiny_skia::LineCap::Round;
            stroke.line_join = tiny_skia::LineJoin::Round;
            self.pixmap.stroke_path(&built, &paint, &stroke, Transform::identity(), None);

            // Inner crisp line.
            paint.set_color(Color::from_rgba8(
                character.contour_color[0],
                character.contour_color[1],
                character.contour_color[2],
                inner_a,
            ));
            stroke.width = inner_w;
            self.pixmap.stroke_path(&built, &paint, &stroke, Transform::identity(), None);
        }
    }

    /// Draw a faint cracked-glass wireframe overlay — fixed lattice
    /// jittered by `time`, just enough to read as "broken hologram".
    /// The full Voronoi cell-relaxation algorithm would be lovely; for
    /// now we approximate with a hexagonal grid which reads the same.
    fn draw_voronoi_overlay(&mut self, mesh: crate::character::VoronoiMesh, time: f32) {
        let w = self.settings.width as f32;
        let h = self.settings.height as f32;
        let mut paint = Paint::default();
        paint.set_color(Color::from_rgba8(
            mesh.color[0],
            mesh.color[1],
            mesh.color[2],
            (mesh.alpha * 255.0) as u8,
        ));
        paint.anti_alias = true;
        let mut stroke = Stroke::default();
        stroke.width = mesh.line_width;
        const STEP: f32 = 38.0;
        let drift = (time * 0.3).sin() * 6.0;
        for row in 0..=((h / STEP) as i32 + 1) {
            let y = row as f32 * STEP * 0.866 + drift;
            let x_off = if row % 2 == 0 { 0.0 } else { STEP * 0.5 };
            for col in -1..=((w / STEP) as i32 + 1) {
                let x = col as f32 * STEP + x_off + (time * 0.5).cos() * 4.0;
                // Three of the six hexagon edges per cell — enough to
                // imply the lattice without doubling every edge.
                let mut pb = PathBuilder::new();
                pb.move_to(x, y);
                pb.line_to(x + STEP * 0.5, y + STEP * 0.288);
                pb.line_to(x + STEP, y);
                pb.line_to(x + STEP, y - STEP * 0.577);
                if let Some(path) = pb.finish() {
                    self.pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
                }
            }
        }
    }
}

/// Premultiplied alpha additive blend onto a BGRA pixel. (`tiny_skia`
/// stores in `RGBA8` byte order but premultiplied — we apply a simple
/// add for the particle pass since the field is over a dark trail.)
#[inline]
fn blend_add_premul(dst: &mut [u8], r: u8, g: u8, b: u8, a: u8) {
    let a_f = a as u32;
    // Scaled contribution.
    let cr = (r as u32 * a_f) / 255;
    let cg = (g as u32 * a_f) / 255;
    let cb = (b as u32 * a_f) / 255;
    dst[0] = (dst[0] as u32 + cr).min(255) as u8;
    dst[1] = (dst[1] as u32 + cg).min(255) as u8;
    dst[2] = (dst[2] as u32 + cb).min(255) as u8;
    dst[3] = (dst[3] as u32 + a_f).min(255) as u8;
}
