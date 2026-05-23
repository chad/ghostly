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

use crate::character::{AccentParticles, Character, ContourBaseline, NebulaCloud};
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
    /// `0..=1` audio loudness — written by the host each frame.
    /// Drives a per-particle reactive jitter so the face *visibly*
    /// shimmers when the agent (or someone else) is speaking. Without
    /// this the face renders motionless except for the breathing
    /// pulse, which reads as a still image.
    pub audio_level: f32,
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
            audio_level: 0.0,
        }
    }

    /// Push the current audio loudness (`0..=1`) into the state.
    /// Renderer reads this each frame to compute per-particle reactive
    /// jitter — the difference between a still face and a *visibly
    /// alive* face that shimmers with speech.
    pub fn set_audio_level(&mut self, level: f32) {
        self.audio_level = level.clamp(0.0, 1.0);
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

        // Nebula cloud goes underneath everything — a dim radial glow
        // behind the face. Skipped when the character doesn't ask for
        // one.
        if let Some(neb) = character.render_config.nebula_cloud {
            self.paint_nebula(neb, cx, cy);
        }

        let breath =
            (time * character.contour_baseline.breath_rate * std::f32::consts::TAU).sin() * 0.5
                + 0.5;
        let breath_pulse = 1.0 + (breath - 0.5) * character.contour_baseline.breath_depth;

        // ── Particle pass ─────────────────────────────────────────
        let data = self.pixmap.data_mut();
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        // Audio reactivity. Idle drift is large enough that a silent
        // face still visibly *crackles* in place; audio kicks add a
        // big extra Lissajous on top. Empirically tuned by watching
        // the avatar reference and matching the felt motion intensity.
        let level = state.audio_level;
        let react = level * 8.0;

        // Whole-face gentle sway — a slow ~0.5 Hz Lissajous of the
        // entire head, so it looks like she's alive and breathing
        // rather than rendered once and pinned. Independent of the
        // per-particle drift; this is gross body motion.
        let sway_x = (time * 0.45).sin() * 0.035 + (time * 0.31 + 1.7).cos() * 0.018;
        let sway_y = (time * 0.52 + 0.8).sin() * 0.022 + (time * 0.27).cos() * 0.012;
        let sway_z = (time * 0.38).sin() * 0.030;

        for (i, p) in state.particles.iter().enumerate() {
            let pos = state.live[i];

            // Per-particle idle drift — three independent multi-axis
            // wobbles per particle so the face never reads as static.
            // Magnitudes ~5× the original (which was barely visible)
            // and a 3rd higher-frequency band gives crackle even when
            // the audio level is zero.
            let nt = time * 0.9 + level * 2.5;
            let idx = i as f32;
            let drift_x = (idx * 0.00097 + nt).sin()
                * (idx * 0.00071 + nt * 0.7).cos()
                * 0.13
                + (idx * 0.013 + time * 2.0).sin() * 0.06 * (0.6 + react)
                + (idx * 0.077 + time * 6.4).sin() * 0.025;
            let drift_y = (idx * 0.00127 + nt * 1.1).cos()
                * (idx * 0.00089 + nt * 0.5).sin()
                * 0.12
                + (idx * 0.017 + time * 2.3).cos() * 0.055 * (0.6 + react)
                + (idx * 0.083 + time * 5.7).cos() * 0.022;
            let drift_z = (idx * 0.00167 + nt * 0.8).sin() * 0.085
                + (idx * 0.021 + time * 1.7).sin() * 0.045 * (0.6 + react)
                + (idx * 0.093 + time * 4.9).sin() * 0.018;

            let x = pos[0] + drift_x + sway_x;
            let y = pos[1] + drift_y + sway_y;
            let z = pos[2] + drift_z + sway_z;

            // Perspective project. Slightly squashed so particles in
            // front read bigger than ones behind.
            let depth = 1.0 / (2.5 + z);
            let sx = cx + x * sc * depth;
            // Canvas Y is inverted (top-left origin) — flip.
            let sy = cy - y * sc * depth;

            // Size grows with materialization + breath + audio. The
            // audio component makes the field shimmer harder as the
            // agent speaks — same trick face-of-god-face.js uses.
            let size = (1.0 + state.pt[i] * 1.6 + level * 1.4)
                * depth
                * breath_pulse;
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

        // ── Accent particle ring (Utopia's lavender motes) ───────
        if let Some(ring) = character.render_config.accent_particles {
            self.draw_accent_particles(ring, time, cx, cy, sc);
        }

        // ── Voronoi-mesh overlay (Oblivion red / Utopia gold) ────
        if let Some(mesh) = character.render_config.voronoi_mesh {
            self.draw_voronoi_overlay(mesh, time);
        }

        &self.pixmap
    }

    /// Paint a soft radial gradient behind the face. Manual loop over
    /// every pixel — cheap at 640×360 and lets us match the avatar
    /// exactly (premultiplied additive, soft falloff).
    fn paint_nebula(&mut self, neb: NebulaCloud, cx: f32, cy: f32) {
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        let max_r = (cx.max(self.settings.width as f32 - cx)
            .powi(2)
            + cy.max(self.settings.height as f32 - cy).powi(2))
        .sqrt();
        let data = self.pixmap.data_mut();
        let nr = neb.color[0];
        let ng = neb.color[1];
        let nb = neb.color[2];
        for py in 0..ph {
            let dy = py as f32 - cy;
            for px in 0..pw {
                let dx = px as f32 - cx;
                let r = (dx * dx + dy * dy).sqrt() / max_r;
                // Soft falloff — squared so the centre is bright and
                // the corners fade gracefully.
                let strength = (1.0 - r).max(0.0).powi(2) * neb.intensity;
                if strength < 0.005 {
                    continue;
                }
                let off = (py * pw + px) * 4;
                let a = (strength * 255.0).min(255.0) as u32;
                let cr = (nr as u32 * a) / 255;
                let cg = (ng as u32 * a) / 255;
                let cb = (nb as u32 * a) / 255;
                data[off] = (data[off] as u32 + cr).min(255) as u8;
                data[off + 1] = (data[off + 1] as u32 + cg).min(255) as u8;
                data[off + 2] = (data[off + 2] as u32 + cb).min(255) as u8;
                // Alpha already 255 from fade_to_black.
            }
        }
    }

    /// Draw an orbiting ring of accent particles. Each particle has a
    /// distinct radius + phase derived from its index, so the ring
    /// reads as a *cloud* rather than a single tracked orbit.
    fn draw_accent_particles(
        &mut self,
        ring: AccentParticles,
        time: f32,
        cx: f32,
        cy: f32,
        sc: f32,
    ) {
        let data = self.pixmap.data_mut();
        let pw = self.settings.width as usize;
        let ph = self.settings.height as usize;
        let r_base = ring.radius * sc * 0.4;
        let a = (ring.alpha * 255.0).min(255.0) as u32;
        let r = ring.color[0];
        let g = ring.color[1];
        let b = ring.color[2];
        for i in 0..ring.count {
            // Deterministic per-particle phase + radius variance.
            let s = i as f32;
            let phase = (s * 0.6180339887).fract() * std::f32::consts::TAU;
            let speed = 0.18 + (s * 0.123).sin().abs() * 0.18;
            let rad = r_base * (0.85 + (s * 0.31).sin().abs() * 0.45);
            let ang = phase + time * speed;
            let px = cx + ang.cos() * rad;
            let py = cy + ang.sin() * rad * 0.85; // squashed vertically
            let pxi = px.floor() as i32;
            let pyi = py.floor() as i32;
            if pxi < 0 || pyi < 0 || (pxi as usize) >= pw || (pyi as usize) >= ph {
                continue;
            }
            // 2×2 splat — gentle and cheap.
            for dy in 0..2 {
                for dx in 0..2 {
                    let xi = pxi + dx;
                    let yi = pyi + dy;
                    if xi < 0 || yi < 0 || (xi as usize) >= pw || (yi as usize) >= ph {
                        continue;
                    }
                    let off = ((yi as usize) * pw + xi as usize) * 4;
                    let cr = (r as u32 * a) / 255;
                    let cg = (g as u32 * a) / 255;
                    let cb = (b as u32 * a) / 255;
                    data[off] = (data[off] as u32 + cr).min(255) as u8;
                    data[off + 1] = (data[off + 1] as u32 + cg).min(255) as u8;
                    data[off + 2] = (data[off + 2] as u32 + cb).min(255) as u8;
                }
            }
        }
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

        let band_glow = character.render_config.band_glow;

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

            let mut paint = Paint::default();
            paint.anti_alias = true;
            let mut stroke = Stroke::default();
            stroke.line_cap = tiny_skia::LineCap::Round;
            stroke.line_join = tiny_skia::LineJoin::Round;

            // Band glow — wider, very faint pass underneath. Gives
            // Utopia's wire bands their luminous halo.
            if band_glow {
                paint.set_color(Color::from_rgba8(
                    character.contour_color[0],
                    character.contour_color[1],
                    character.contour_color[2],
                    outer_a / 5,
                ));
                stroke.width = outer_w * 2.4;
                self.pixmap.stroke_path(&built, &paint, &stroke, Transform::identity(), None);
            }

            // Outer glow — softer + wider.
            paint.set_color(Color::from_rgba8(
                character.contour_color[0],
                character.contour_color[1],
                character.contour_color[2],
                outer_a / 2,
            ));
            stroke.width = outer_w;
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

    /// Cracked-glass wireframe overlay. A deterministic Poisson-disk
    /// scatter of seed points (gives an organic, non-grid feel), then
    /// edges between any two seeds within a threshold distance — the
    /// proximity graph reads visually identically to a true Voronoi
    /// triangulation at this density. Each seed wobbles around its
    /// rest position over time so the mesh subtly *breathes* like a
    /// holographic lattice rather than sitting flat.
    ///
    /// Skipping the true Lloyd-relaxed Voronoi cells because at ~80
    /// seeds the proximity graph is indistinguishable to the eye but
    /// is O(N²) instead of needing a full geometric kernel. If we
    /// ever raise the density past ~200 seeds we'll need the real
    /// thing.
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

        // ── Deterministic Poisson-disk-ish seed scatter ──
        // We use the golden-ratio low-discrepancy sequence + a quick
        // rejection step to keep seeds at least `MIN_SEP` apart. Same
        // input → same lattice, so the mesh is stable across frames.
        const TARGET_SEEDS: usize = 90;
        const PHI: f32 = 1.618_034;
        const MIN_SEP: f32 = 44.0;
        let mut seeds: Vec<[f32; 2]> = Vec::with_capacity(TARGET_SEEDS);
        let mut tried = 0;
        // The bg lattice runs slightly larger than the canvas so the
        // mesh fades off the edges rather than ending abruptly.
        let pad = MIN_SEP;
        while seeds.len() < TARGET_SEEDS && tried < TARGET_SEEDS * 8 {
            let i = tried as f32;
            let u = (i * PHI).fract();
            let v = ((i * PHI).fract() * PHI).fract();
            let x = -pad + u * (w + pad * 2.0);
            let y = -pad + v * (h + pad * 2.0);
            let ok = seeds.iter().all(|&[sx, sy]| {
                let dx = sx - x;
                let dy = sy - y;
                dx * dx + dy * dy >= MIN_SEP * MIN_SEP
            });
            if ok {
                seeds.push([x, y]);
            }
            tried += 1;
        }

        // ── Time-modulated wobble ──
        // Each seed drifts on a per-seed Lissajous so the lattice
        // shivers without losing structure.
        let live: Vec<[f32; 2]> = seeds
            .iter()
            .enumerate()
            .map(|(i, [x, y])| {
                let phase = i as f32 * 0.9;
                let amp = MIN_SEP * 0.12;
                let dx = (time * 0.4 + phase).sin() * amp;
                let dy = (time * 0.55 + phase * 1.3).cos() * amp * 0.7;
                [x + dx, y + dy]
            })
            .collect();

        // ── Proximity-graph edges ──
        // For each pair within `EDGE_MAX`, stroke a line. The cutoff
        // is chosen so each seed connects to ~5-6 neighbours on
        // average — same connectivity as a typical Voronoi cell.
        const EDGE_MAX: f32 = 78.0;
        let edge_max_sq = EDGE_MAX * EDGE_MAX;
        let mut pb = PathBuilder::new();
        for i in 0..live.len() {
            let [x1, y1] = live[i];
            for &[x2, y2] in &live[i + 1..] {
                let dx = x2 - x1;
                let dy = y2 - y1;
                let d2 = dx * dx + dy * dy;
                if d2 < edge_max_sq {
                    pb.move_to(x1, y1);
                    pb.line_to(x2, y2);
                }
            }
        }
        if let Some(path) = pb.finish() {
            self.pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
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
